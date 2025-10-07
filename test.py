import torch
import kornia
import cv2
import argparse
import config
import numpy as np
import os
import glob
import pandas as pd
import time
from models.builder import build_model
from utils.coordinate_transforms import create_cartesian_from_polar_maps, remap_image, create_polar_from_cartesian_maps
from evaluation.metrics import calculate_metrics
from utils.masking import apply_gray_border

def deblur_image(input_path, output_path, model, device, cart_map_x, cart_map_y, polar_to_cart_map_x, polar_to_cart_map_y, blur_angle, sharp_path=None):
    """
    Deblurs a single image.
    
    Args:
        sharp_path: Path to the corresponding sharp/ground truth image for metric calculation (optional)
    
    Returns:
        Tuple: (success: bool, metrics: dict or None)
    """
    # Load and prepare image
    input_img_bgr = cv2.imread(input_path)
    if input_img_bgr is None:
        print(f"Warning: Could not read image at {input_path}")
        return False, None

    # Resize to the dimensions the model was trained on
    input_img_bgr = cv2.resize(input_img_bgr, (config.IMG_WIDTH, config.IMG_HEIGHT))
    
    # Convert to tensor, normalize, and add batch dimension
    input_img_rgb = cv2.cvtColor(input_img_bgr, cv2.COLOR_BGR2RGB)
    input_tensor = torch.from_numpy(input_img_rgb).permute(2, 0, 1).float().to(device) / 255.0
    input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension, shape is now (1, C, H, W)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    # Perform inference
    with torch.no_grad():
        # The maps are (1, H, W). The image is (1, C, H, W).
        # Expand the maps to match the image's batch size.
        batch_size = input_tensor.shape[0]

        # Convert image to polar coordinates
        b_cart_map_x = cart_map_x.to(device).expand(batch_size, -1, -1)
        b_cart_map_y = cart_map_y.to(device).expand(batch_size, -1, -1)

        polar_blurry_tensor = remap_image(input_tensor, b_cart_map_x, b_cart_map_y)
        
        if config.MODEL_NAME == "non_blind_iterative_rmd":
            blur_angle_deg = torch.tensor([blur_angle], device=device)
            center_x_rel = torch.tensor([0.5], device=device)  # Center of image
            center_y_rel = torch.tensor([0.5], device=device)  # Center of image
            
            stage_outputs, _ = model(
                polar_blurry_tensor,
                blur_angle_deg,
                center_x_rel,
                center_y_rel
            )

        # Get the final stage output
        final_polar_output = stage_outputs[-1]
        
        # Convert the result back to Cartesian, expanding maps to match batch size
        b_polar_to_cart_map_x = polar_to_cart_map_x.to(device).expand(batch_size, -1, -1)
        b_polar_to_cart_map_y = polar_to_cart_map_y.to(device).expand(batch_size, -1, -1)
        final_cart_output = remap_image(final_polar_output, b_polar_to_cart_map_x, b_polar_to_cart_map_y)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    end_time = time.time()
    inference_time = end_time - start_time
    print(f"Inference time for one image: {inference_time:.6f} seconds")

    # Apply the same circular mask and gray border as the plotter
    dummy_sharp = torch.zeros_like(final_cart_output)
    deblurred_bordered, _ = apply_gray_border(final_cart_output, dummy_sharp)

    # Convert tensor back to a displayable image
    output_tensor_no_batch = deblurred_bordered.squeeze(0).cpu().detach()
    output_tensor_clamped = torch.clamp(output_tensor_no_batch, 0, 1)
    output_np = output_tensor_clamped.permute(1, 2, 0).numpy()
    output_np = (output_np * 255).astype(np.uint8)
    output_image_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, output_image_bgr)
    
    # Calculate metrics if sharp image path is provided
    metrics_result = None
    if sharp_path and os.path.isfile(sharp_path):
        # Load sharp image
        sharp_img_bgr = cv2.imread(sharp_path)
        if sharp_img_bgr is not None:
            # Resize sharp image to match model dimensions
            sharp_img_bgr = cv2.resize(sharp_img_bgr, (config.IMG_WIDTH, config.IMG_HEIGHT))
            sharp_img_rgb = cv2.cvtColor(sharp_img_bgr, cv2.COLOR_BGR2RGB)
            sharp_tensor = torch.from_numpy(sharp_img_rgb).permute(2, 0, 1).float().to(device) / 255.0
            sharp_tensor = sharp_tensor.unsqueeze(0) # Add batch dimension
            
            # Calculate metrics using the same function as the trainer
            metrics_result = calculate_metrics(final_cart_output, sharp_tensor)
            print(f"Metrics - PSNR: {metrics_result['psnr']:.2f} dB, SSIM: {metrics_result['ssim']:.4f}")
        else:
            print(f"Warning: Could not read sharp image at {sharp_path}")
    
    return True, metrics_result, inference_time

def process_multiple_images(args):
    """
    Processes multiple images based on input arguments.
    """
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # Load Model
    model = build_model(config)
    print(f"Loading model weights from: {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, weights_only=True, map_location=device))
    model.to(device)
    model.eval()

    # Generate Coordinate Maps directly instead of loading them
    print("Generating coordinate transformation maps...")
    polar_to_cart_map_x, polar_to_cart_map_y = create_cartesian_from_polar_maps(
        config.IMG_HEIGHT, config.IMG_WIDTH
    )
    cart_map_x, cart_map_y = create_polar_from_cartesian_maps(
        config.IMG_HEIGHT, config.IMG_WIDTH
    )

    # CSV Processing Logic
    if args.csv_path:
        if not os.path.isfile(args.csv_path):
            print(f"Error: CSV file not found at {args.csv_path}")
            return
        if not os.path.isdir(args.output):
            print("Error: For CSV processing, --output must be a directory.")
            return
        
        print(f"Reading image data from CSV: {args.csv_path}")
        df = pd.read_csv(args.csv_path)
        
        # Check for required columns
        required_cols = ['blurred_path', 'blur_angle']
        if not all(col in df.columns for col in required_cols):
            print(f"Error: CSV must contain the columns: {required_cols}")
            return

        # Check if original_path column exists for metrics calculation
        has_sharp_paths = 'original_path' in df.columns

        os.makedirs(args.output, exist_ok=True)
        successful = 0
        all_metrics = []
        all_inference_times = []
        
        # Get the directory of the CSV file to resolve relative paths
        csv_dir = os.path.dirname(args.csv_path)
        # The paths in the CSV are relative to the parent of the CSV's directory
        base_dir = os.path.dirname(csv_dir)
        
        for index, row in df.iterrows():
            # Construct the path from the parent directory
            input_path = os.path.join(base_dir, row['blurred_path'].replace('/', os.sep))
            blur_angle = row['blur_angle']
            
            # Get original path if available
            original_path = None
            if has_sharp_paths and pd.notna(row['original_path']):
                original_path = os.path.join(base_dir, row['original_path'].replace('/', os.sep))

            print(f"Found: {input_path} with angle {blur_angle:.2f}")

            if not os.path.isfile(input_path):
                print(f"Warning: Skipping missing file: {input_path}")
                continue

            filename = os.path.basename(input_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(args.output, f"{name}_deblurred{ext}")
            
            print(f"Processing: {filename} with angle {blur_angle:.2f}")
            success, metrics, inf_time = deblur_image(input_path, output_path, model, device,
                                          cart_map_x, cart_map_y, polar_to_cart_map_x, polar_to_cart_map_y,
                                          blur_angle, original_path)
            if success:
                successful += 1
                all_inference_times.append(inf_time) # Collect inference time
                if metrics:
                    all_metrics.append(metrics)
        
        print(f"\nSuccessfully processed {successful}/{len(df)} images from CSV.")
        print(f"Results saved to: {args.output}")
        
        # Print average metrics if available
        if all_metrics:
            avg_psnr = sum(m['psnr'] for m in all_metrics) / len(all_metrics)
            avg_ssim = sum(m['ssim'] for m in all_metrics) / len(all_metrics)
            print(f"Average Metrics - PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

        # Calculate and print average inference time
        if all_inference_times:
            # The first run can be slower (warm-up), so it's often excluded from averages.
            if len(all_inference_times) > 1:
                avg_time = sum(all_inference_times[1:]) / (len(all_inference_times) - 1)
                print(f"Average Inference Time (excluding first image): {avg_time:.6f} seconds")
            else:
                avg_time = sum(all_inference_times) / len(all_inference_times)
                print(f"Average Inference Time: {avg_time:.6f} seconds")
        
        return

    # Determine input files
    input_files = []
    if os.path.isfile(args.input):
        # Single file
        input_files = [args.input]
    elif os.path.isdir(args.input):
        # Directory - find all image files
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        for ext in extensions:
            input_files.extend(glob.glob(os.path.join(args.input, ext)))
            input_files.extend(glob.glob(os.path.join(args.input, ext.upper())))
        input_files.sort()
    else:
        # Pattern matching
        input_files = glob.glob(args.input)
        input_files.sort()

    if not input_files:
        print(f"No image files found matching: {args.input}")
        return

    print(f"Found {len(input_files)} image(s) to process")

    # Determine output strategy
    if len(input_files) == 1:
        # Single input file
        if os.path.isdir(args.output):
            # Output is directory, create filename
            filename = os.path.basename(input_files[0])
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(args.output, f"{name}_deblurred{ext}")
        else:
            # Output is specific file path
            output_path = args.output
        
        print(f"Processing: {input_files[0]} -> {output_path}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success, metrics, inf_time = deblur_image(input_files[0], output_path, model, device, 
                                      cart_map_x, cart_map_y, polar_to_cart_map_x, polar_to_cart_map_y, 
                                      args.blur_angle, args.sharp_path)
        if success:
            print(f"Successfully saved deblurred image to: {output_path}")
        else:
            print(f"Failed to process: {input_files[0]}")
    else:
        # Multiple input files
        if not os.path.isdir(args.output):
            print("For multiple input files, output must be a directory")
            return
        
        os.makedirs(args.output, exist_ok=True)
        
        successful = 0
        all_metrics = []
        all_inference_times = []
        for input_path in input_files:
            filename = os.path.basename(input_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(args.output, f"{name}_deblurred{ext}")
            
            print(f"Processing: {filename}")
            success, metrics, inf_time = deblur_image(input_path, output_path, model, device,
                                          cart_map_x, cart_map_y, polar_to_cart_map_x, polar_to_cart_map_y,
                                          args.blur_angle)
            if success:
                successful += 1
                all_inference_times.append(inf_time) # Collect inference time
                if metrics:
                    all_metrics.append(metrics)
            
        print(f"Successfully processed {successful}/{len(input_files)} images")
        print(f"Results saved to: {args.output}")
        
        # Print average metrics if available
        if all_metrics:
            avg_psnr = sum(m['psnr'] for m in all_metrics) / len(all_metrics)
            avg_ssim = sum(m['ssim'] for m in all_metrics) / len(all_metrics)
            print(f"Average Metrics - PSNR: {avg_psnr:.2f} dB, SSIM: {avg_ssim:.4f}")

        # Calculate and print average inference time
        if all_inference_times:
            # The first run can be slower (warm-up), so it's often excluded from averages.
            if len(all_inference_times) > 1:
                avg_time = sum(all_inference_times[1:]) / (len(all_inference_times) - 1)
                print(f"Average Inference Time (excluding first image): {avg_time:.6f} seconds")
            else:
                avg_time = sum(all_inference_times) / len(all_inference_times)
                print(f"Average Inference Time: {avg_time:.6f} seconds")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deblur image(s) using a pre-trained model.")
    parser.add_argument(
        '--input', 
        type=str, 
        default=None, 
        help="Path to input image, directory of images, or glob pattern. Use --csv_path for CSV-based processing."
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True, 
        help="Output path: specific file for single image, or directory for multiple images/CSV processing"
    )
    parser.add_argument(
        '--csv_path',
        type=str,
        default=None,
        help="Path to a CSV file with 'blurred_path' and 'blur_angle' columns for batch processing. Optionally include 'sharp_path' for metrics."
    )
    parser.add_argument(
        '--model_path', 
        type=str, 
        default=config.MODEL_SAVE_PATH, 
        help=f"Path to the trained model weights file (.pth). Defaults to config: {config.MODEL_SAVE_PATH}"
    )
    parser.add_argument(
        '--blur_angle', 
        type=float, 
        default=30.0, 
        help="Estimated blur angle in degrees (default: 30.0). Not used if --csv_path is provided."
    )
    parser.add_argument(
        '--sharp_path',
        type=str,
        default=None,
        help="Path to the corresponding sharp/ground truth image for metric calculation (single image mode only)."
    )

    args = parser.parse_args()

    if not args.input and not args.csv_path:
        parser.error("Either --input (for a single file/directory) or --csv_path must be provided.")
    if args.input and args.csv_path:
        parser.error("Provide either --input or --csv_path, not both.")

    process_multiple_images(args)