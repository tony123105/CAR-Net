import torch
from torch.utils.data import random_split
import config
import numpy as np
import random
from data_handling.dataset import DeblurDataset
from models.builder import build_model
from core.trainer import Trainer
from core.losses import CombinedLoss, PhysicsLoss, L1ReconstructionLoss, SSIMLoss

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure that all operations are deterministic on GPU (if used) for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():
    # Set seed for reproducibility
    set_seed(config.SEED)
    print(f"--- Using seed: {config.SEED} ---")

    # Setup device
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Data
    full_dataset = DeblurDataset(
        params_file=config.PARAMS_FILE,
        device=device,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH
    )

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    print(f"Dataset split into {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")

    model = build_model(config).to(device)
    
    # Define Loss Strategy
    loss_function = CombinedLoss(
        physics_weight=1.0, 
        recon_weight=1.0, 
        ssim_weight=0.5,
        angle_weight=0.1 if config.RMD_MODEL_CONFIG['use_angle_correction'] else 0.0,
        max_angle=config.MAX_BLUR_ANGLE,
        n_steps_physics=config.BLUR_N_STEPS
    )
    print("--- Loss Function Initialized ---")
    print(f"Weights -> Reconstruction: {loss_function.recon_weight}, SSIM: {loss_function.ssim_weight}, Physics: {loss_function.physics_weight}, Angle: {loss_function.angle_weight}")


    # Initialize and run the trainer
    trainer = Trainer(
        model=model, 
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        loss_function=loss_function,
        config=config
    )
    trainer.train()
    trainer.evaluate_and_save()

if __name__ == '__main__':
    main()