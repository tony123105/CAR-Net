import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data_handling.dataset import DeblurDataset
from utils.coordinate_transforms import create_cartesian_from_polar_maps, remap_image
from evaluation import plotter, metrics
import time
import os
import logging

class Trainer:
    def __init__(self, model: nn.Module, train_dataset, val_dataset, loss_function, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.loss_function = loss_function
        self.config = config
        self.device = config.DEVICE

        # Setup Logger 
        os.makedirs(config.OUTPUT_DIR, exist_ok=True) # Ensure output dir exists
        log_file_path = os.path.join(config.OUTPUT_DIR, f"{config.MODEL_NAME}_training_log.txt")
        
        self.logger = logging.getLogger(f"Trainer_{config.MODEL_NAME}")
        self.logger.setLevel(logging.INFO)
        
        if not self.logger.handlers:
            # File handler
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
            self.logger.addHandler(file_handler)
            
            # Console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(console_handler)

        self.train_loader = DataLoader(self.train_dataset, batch_size=config.BATCH_SIZE, shuffle=True) # Dataloader for training
        self.val_loader = DataLoader(self.val_dataset, batch_size=config.BATCH_SIZE, shuffle=False) # Dataloader for validation
        
        # Create a separate test dataset and loader
        test_dataset = DeblurDataset(
            params_file=config.TEST_PARAMS_FILE,
            device=self.device,
            img_height=config.TEST_IMG_HEIGHT,
            img_width=config.TEST_IMG_WIDTH
        )
        self.test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
        self.logger.info(f"Loaded {len(test_dataset)} samples for final evaluation from '{config.TEST_PARAMS_FILE}'")

        # Handle optimizer creation only if there are trainable parameters
        model_params = list(self.model.parameters())
        if model_params:
            self.optimizer = torch.optim.Adam(model_params, lr=config.LEARNING_RATE)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
            self.logger.info("Optimizer and scheduler created for trainable parameters.")
        else:
            self.optimizer = None
            self.scheduler = None
            self.logger.warning(
                "The model has no trainable parameters. "
                "Training will be skipped. Proceeding to evaluation."
            )

        self.polar_to_cart_map_x, self.polar_to_cart_map_y = create_cartesian_from_polar_maps(
            config.IMG_HEIGHT, config.IMG_WIDTH
        )
        self.loss_history = []
        self.detailed_loss_history = []
        self.val_loss_history = []
        self.best_val_loss = float('inf')

        # Create a directory for epoch checkpoints
        self.checkpoint_dir = os.path.join(os.path.dirname(config.MODEL_SAVE_PATH), "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.logger.info(f"Epoch checkpoints will be saved in '{self.checkpoint_dir}'")

    def train(self):
        # Skip training if there's no optimizer (i.e., no trainable parameters)
        if not self.optimizer:
            self.logger.info("--- Skipping Training: No trainable parameters found. ---")
            return
        self.logger.info("--- Starting Training ---")
        start_time = time.time()

        for epoch in range(self.config.NUM_EPOCHS):
            self.model.train()
            train_loss = 0
            epoch_detailed_losses = {}

            for i, data in enumerate(self.train_loader):
                blurry_polar = data["blurry_polar"]
                batch_size = blurry_polar.shape[0]
                
                if self.config.MODEL_NAME == "car-net":
                    stage_outputs_polar, residual_angle = self.model(
                        blurry_polar,
                        data["blur_angle_deg"],
                        data["center_x_rel"],
                        data["center_y_rel"]
                    )
                    estimated_angle = residual_angle
                    #angle_target = torch.zeros_like(residual_angle)

                # Multi-Stage Supervision: Convert all polar stages to cartesian
                map_x = self.polar_to_cart_map_x.to(self.device).expand(batch_size, -1, -1)
                map_y = self.polar_to_cart_map_y.to(self.device).expand(batch_size, -1, -1)
                stage_outputs_cartesian = [remap_image(p, map_x, map_y) for p in stage_outputs_polar]
                final_output_cartesian = stage_outputs_cartesian[-1]

                loss_inputs = {
                    "stage_outputs_polar": stage_outputs_polar,
                    "stage_outputs_cartesian": stage_outputs_cartesian, # Pass all cartesian stages
                    "original_blurry_polar": data["blurry_polar"],
                    "blurry_cartesian": data["blurry_cartesian"],
                    "sharp_polar": data["sharp_polar"],
                    "sharp_cartesian": data["sharp_cartesian"],
                    "blur_angle_deg": data["original_blur_angle"], # Use the correct target for the loss
                    "center_x_rel": data["center_x_rel"],
                    "center_y_rel": data["center_y_rel"],
                    "estimated_angle": estimated_angle
                }

                total_loss, loss_dict = self.loss_function(**loss_inputs)
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += total_loss.item()
                
                # Accumulate the detailed losses from the batch
                for key, value in loss_dict.items():
                    if key not in epoch_detailed_losses:
                        epoch_detailed_losses[key] = 0.0
                    epoch_detailed_losses[key] += value

            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in self.val_loader:

                    blurry_polar = data["blurry_polar"]
                    batch_size = blurry_polar.shape[0]

                    if self.config.MODEL_NAME == "car-net":
                        stage_outputs_polar, residual_angle = self.model(
                            blurry_polar,
                            data["blur_angle_deg"],
                            data["center_x_rel"],
                            data["center_y_rel"]
                        )
                        estimated_angle = residual_angle
                        #angle_target = torch.zeros_like(residual_angle)

                    # Multi-Stage Supervision: Convert all polar stages to cartesian for validation 
                    map_x = self.polar_to_cart_map_x.to(self.device).expand(batch_size, -1, -1)
                    map_y = self.polar_to_cart_map_y.to(self.device).expand(batch_size, -1, -1)
                    stage_outputs_cartesian = [remap_image(p, map_x, map_y) for p in stage_outputs_polar]
                    final_output_cartesian = stage_outputs_cartesian[-1]
                    
                    loss_inputs = {
                        "stage_outputs_polar": stage_outputs_polar,
                        "stage_outputs_cartesian": stage_outputs_cartesian, # Pass all cartesian stages
                        "original_blurry_polar": data["blurry_polar"],
                        "blurry_cartesian": data["blurry_cartesian"],
                        "sharp_polar": data["sharp_polar"],
                        "sharp_cartesian": data["sharp_cartesian"],
                        "blur_angle_deg": data["original_blur_angle"], # Use the correct target for the loss
                        "center_x_rel": data["center_x_rel"],
                        "center_y_rel": data["center_y_rel"],
                        "estimated_angle": estimated_angle
                    }
                    
                    total_val_loss, _ = self.loss_function(**loss_inputs)
                    val_loss += total_val_loss.item()

            avg_train_loss = train_loss / len(self.train_loader)
            avg_val_loss = val_loss / len(self.val_loader)
            
            # Average the detailed losses and append all losses to history
            avg_detailed_losses = {key: value / len(self.train_loader) for key, value in epoch_detailed_losses.items()}
            self.loss_history.append(avg_train_loss)
            self.val_loss_history.append(avg_val_loss)
            self.detailed_loss_history.append(avg_detailed_losses)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"Epoch [{epoch+1}/{self.config.NUM_EPOCHS}], Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}, LR: {current_lr}")

            # Save the model if it has the best validation loss
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), self.config.MODEL_SAVE_PATH)
                self.logger.info(f"  -> New best model saved with validation loss: {avg_val_loss:.6f}")

            # Save a checkpoint for every epoch with elapsed time
            elapsed_epoch_time = time.time() - start_time
            epoch_save_path = os.path.join(self.checkpoint_dir, f"model_epoch_{epoch+1}_time_{elapsed_epoch_time:.2f}s.pth")
            torch.save(self.model.state_dict(), epoch_save_path)

            self.scheduler.step(avg_val_loss)

        elapsed_time = time.time() - start_time
        self.logger.info(f"--- Training Finished in {elapsed_time:.2f}s ---")

    def evaluate_and_save(self):
        self.logger.info("--- Evaluating and Saving Results on Test Set ---")
        # If there was no training, we evaluate the initial state of the model
        # without trying to load a saved checkpoint.
        if self.optimizer:
            # Load the best model state for final evaluation if training occurred
            self.logger.info(f"Loading best model from '{self.config.MODEL_SAVE_PATH}' for final evaluation.")
            self.model.load_state_dict(torch.load(self.config.MODEL_SAVE_PATH, weights_only=True, map_location=self.device))
        else:
            self.logger.info("Evaluating the model without loading a trained checkpoint.")
        self.model.eval()
        
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)

        # Create coordinate maps specifically for the test set dimensions
        test_map_x, test_map_y = create_cartesian_from_polar_maps(
            self.config.TEST_IMG_HEIGHT, self.config.TEST_IMG_WIDTH
        )
        test_map_x = test_map_x.to(self.device)
        test_map_y = test_map_y.to(self.device)

        total_psnr = 0
        total_ssim = 0
        num_batches = 0
        total_samples = 0 # Track total number of images
        num_saved_samples = 0 # Counter for saved samples

        with torch.no_grad():
            for i, data in enumerate(self.test_loader):
                blurry_polar = data["blurry_polar"].to(self.device)
                batch_size = blurry_polar.shape[0]
                residual_angle = None # Initialize residual_angle

                if self.config.MODEL_NAME == "car-net":
                    stage_outputs, residual_angle = self.model(
                        blurry_polar,
                        data["blur_angle_deg"],
                        data["center_x_rel"],
                        data["center_y_rel"]
                    )

                final_output_polar = stage_outputs[-1]

                # Convert to cartesian for saving and metrics
                map_x = test_map_x.expand(batch_size, -1, -1)
                map_y = test_map_y.expand(batch_size, -1, -1)
                final_output_cartesian = remap_image(final_output_polar, map_x, map_y)

                metrics_dict = metrics.calculate_metrics(final_output_cartesian, data["sharp_cartesian"].to(self.device))
                # Weight the batch metrics by the number of samples in the batch
                total_psnr += metrics_dict["psnr"] * batch_size
                total_ssim += metrics_dict["ssim"] * batch_size
                total_samples += batch_size
                num_batches += 1

                # Save comparison plots for a specified number of samples
                if num_saved_samples < self.config.NUM_TEST_SAVE_RESULTS:
                    for j in range(batch_size):
                        if num_saved_samples < self.config.NUM_TEST_SAVE_RESULTS:
                            # Get single images from the batch
                            blurry_single = data["blurry_cartesian"][j:j+1]
                            deblurred_single = final_output_cartesian[j:j+1]
                            sharp_single = data["sharp_cartesian"][j:j+1].to(self.device)

                            # Calculate metrics for the single image
                            single_metrics = metrics.calculate_metrics(deblurred_single, sharp_single)
                            # Add blur angle to the metrics dict for plotting
                            single_metrics['blur_angle'] = data['blur_angle_deg'][j].item()
                            if residual_angle is not None:
                                single_metrics['residual_angle'] = residual_angle[j].item()

                            plotter.plot_results(
                                blurry_img=blurry_single,
                                deblurred_img=deblurred_single,
                                sharp_img=sharp_single,
                                scores=single_metrics,
                                output_path=os.path.join(self.config.OUTPUT_DIR, f"{self.config.MODEL_NAME}_test_result_{num_saved_samples}.png")
                            )
                            num_saved_samples += 1
                        else:
                            break # Stop saving if the desired number is reached

        # Calculate and print average metrics after the loop
        avg_psnr = total_psnr / total_samples if total_samples > 0 else 0
        avg_ssim = total_ssim / total_samples if total_samples > 0 else 0
        self.logger.info(f"--- Average Test Metrics ---")
        self.logger.info(f"PSNR: {avg_psnr:.2f}")
        self.logger.info(f"SSIM: {avg_ssim:.4f}")

        # Only log training summary if training was performed
        if self.optimizer:
            plotter.plot_loss(
                loss_history=self.loss_history,
                val_loss_history=self.val_loss_history,
                detailed_loss_history=self.detailed_loss_history,
                output_path=os.path.join(self.config.OUTPUT_DIR, f"{self.config.MODEL_NAME}_loss_plot.png")
            )
        

            self.logger.info(f"Best model was saved to '{self.config.MODEL_SAVE_PATH}' with validation loss {self.best_val_loss:.6f} and training loss {self.loss_history[self.val_loss_history.index(self.best_val_loss)]:.6f}")
