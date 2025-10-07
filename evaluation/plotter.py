import torch
import kornia
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List
from utils.masking import apply_circular_mask, apply_gray_border

def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    """Converts a (C, H, W) tensor to a (H, W, C) numpy array for plotting."""
    # Ensure tensor is on CPU and detach it from the computation graph
    image_np = tensor.detach().cpu()
    # Permute from (C, H, W) to (H, W, C)
    image_np = image_np.permute(1, 2, 0).numpy()
    # Denormalize from [0, 1] to [0, 255] and convert to uint8
    image_np = (image_np * 255).astype(np.uint8)
    return image_np

def plot_loss(
    loss_history: List[float], 
    val_loss_history: List[float],
    detailed_loss_history: List[dict], 
    output_path: str
):
    """Saves a plot of the training and validation loss history."""
    plt.figure(figsize=(12, 6))
    
    # Plot total training loss
    plt.plot(loss_history, label="Total Training Loss", linewidth=2, color='blue')

    # Plot total validation loss
    if val_loss_history:
        plt.plot(val_loss_history, label="Total Validation Loss", linewidth=2, color='red')

    # Process and plot detailed training losses if available
    if detailed_loss_history:
        # Convert list of dicts to dict of lists for easier plotting
        detailed_losses = {key: [d[key] for d in detailed_loss_history] for key in detailed_loss_history[0]}
        
        for name, values in detailed_losses.items():
            # Plot each component with a dashed line
            plt.plot(values, label=f"{name} Loss (Train)", linestyle='--')

    plt.title("Training and Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved loss plot to '{output_path}'")

def plot_results(blurry_img, deblurred_img, sharp_img, output_path, scores):
    """
    Plots the blurry input, deblurred output, and sharp ground truth for comparison.
    Uses the same masking approach as metrics calculation.
    """
    # Apply the same gray border and circular masking as used in metrics
    deblurred_bordered, sharp_bordered = apply_gray_border(deblurred_img, sharp_img)

    # For blurry image, apply only circular mask
    blurry_masked = apply_circular_mask(blurry_img[0], mask_color=0.5)

    # Take the first image from the batch for plotting
    blurry_np = tensor_to_image(blurry_masked)
    deblurred_np = tensor_to_image(deblurred_bordered[0])
    sharp_np = tensor_to_image(sharp_bordered[0])

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor('black')

    # Define text properties for white color
    text_props = {'color': 'white', 'fontsize': 16}

    # Plot Blurry Input
    axes[0].imshow(blurry_np, cmap='gray', vmin=0, vmax=255)
    title_str_blurry = f"Blurry Input (Angle: {scores.get('blur_angle', 'N/A'):.1f}°)"
    axes[0].set_title(title_str_blurry, **text_props)
    axes[0].axis('off')

    # Plot Deblurred
    axes[1].imshow(deblurred_np)
    title_str = f"Deblurred (PSNR: {scores['psnr']:.2f}, SSIM: {scores['ssim']:.4f})"
    if 'residual_angle' in scores:
        title_str += f"\nDetected Angle: {scores['residual_angle']:.2f}°"
    axes[1].set_title(title_str, **text_props)
    axes[1].axis('off')

    # Plot Sharp Ground Truth
    axes[2].imshow(sharp_np)
    axes[2].set_title('Ground Truth', **text_props)
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, facecolor='black')
    plt.close(fig)
    print(f"Saved comparison plot to {output_path}")

def plot_psf(psf_tensor: torch.Tensor, output_path: str, title: str = "Point Spread Function (PSF)"):
    """
    Saves a visualization of the given Point Spread Function (PSF).
    Uses a logarithmic scale to make sparse features more visible.
    """
    # Ensure tensor is on CPU and detached from graph
    psf_tensor = psf_tensor.detach().cpu()

    # If the PSF has a channel dimension, handle it
    if psf_tensor.dim() > 2 and psf_tensor.shape[0] in [1, 3]:
        if psf_tensor.shape[0] == 3:
            # Convert RGB to grayscale for visualization
            psf_tensor = kornia.color.rgb_to_grayscale(psf_tensor.unsqueeze(0)).squeeze(0)
        psf_tensor = psf_tensor.squeeze(0)
    
    psf_np = psf_tensor.numpy()


    psf_log = np.log1p(psf_np) 

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # Set background to a dark blue color
    fig.patch.set_facecolor('#000022') # A dark blue
    text_props = {'color': 'white', 'fontsize': 16}

    # Use a more vibrant colormap
    im = ax.imshow(psf_log, cmap='viridis') 
    ax.set_title(title, **text_props)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved PSF visualization to {output_path}")