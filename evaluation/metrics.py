import torch
import kornia.metrics as metrics
from typing import Dict
from utils.masking import apply_circular_mask

def create_circular_mask(height: int, width: int, device: torch.device) -> torch.Tensor:
    """Create a circular mask identifying the deblurred region - consistent with masking.py"""
    center_x, center_y = width // 2, height // 2
    radius = min(height, width) // 2
    
    y, x = torch.meshgrid(torch.arange(height, device=device), torch.arange(width, device=device), indexing='ij')
    distance = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    circular_mask = distance <= radius
    
    return circular_mask

def calculate_metrics_single(pred_tensor: torch.Tensor, target_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Calculates PSNR and SSIM for a single image on the circular deblurred region only.
    """
    channels, height, width = pred_tensor.shape
    
    # Create circular mask for the deblurred region
    circular_mask = create_circular_mask(height, width, pred_tensor.device)
    
    # Calculate both PSNR and SSIM only on circular pixels
    pred_masked = pred_tensor[:, circular_mask]  # Shape: [C, num_pixels_in_circle]
    target_masked = target_tensor[:, circular_mask]  # Shape: [C, num_pixels_in_circle]
    
    mse = torch.mean((pred_masked - target_masked) ** 2)
    if mse > 0:
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    else:
        psnr = torch.tensor(float('inf'))
    
    # SSIM calculation - crop to bounding box and only include circular region
    rows = torch.any(circular_mask, dim=1)
    cols = torch.any(circular_mask, dim=0)
    min_row, max_row = torch.where(rows)[0][[0, -1]]
    min_col, max_col = torch.where(cols)[0][[0, -1]]
    
    # Crop to bounding box
    pred_cropped = pred_tensor[:, min_row:max_row+1, min_col:max_col+1]
    target_cropped = target_tensor[:, min_row:max_row+1, min_col:max_col+1]
    mask_cropped = circular_mask[min_row:max_row+1, min_col:max_col+1]
    
    # Set pixels outside circle to a neutral value, but importantly both images get same treatment
    pred_masked_crop = pred_cropped.clone()
    target_masked_crop = target_cropped.clone()
    pred_masked_crop[:, ~mask_cropped] = 0.5  # Use neutral gray (0.5) instead of black (0.0)
    target_masked_crop[:, ~mask_cropped] = 0.5
    
    # Calculate SSIM on the cropped region
    ssim_val = metrics.ssim(pred_masked_crop.unsqueeze(0), target_masked_crop.unsqueeze(0), window_size=11)
    ssim_scalar = ssim_val.mean()
    
    return {
        'psnr': psnr.item(),
        'ssim': ssim_scalar.item()
    }

def calculate_metrics(pred_tensor: torch.Tensor, target_tensor: torch.Tensor) -> Dict[str, float]:
    """
    Calculates PSNR and SSIM for a batch of images on the circular deblurred region only.
    Processes each image individually and returns the average.

    Args:
        pred_tensor (torch.Tensor): The predicted images (B, C, H, W).
        target_tensor (torch.Tensor): The ground truth sharp images (B, C, H, W).

    Returns:
        A dictionary containing the calculated average 'psnr' and 'ssim'.
    """
    # Ensure batch sizes match before calculation
    if pred_tensor.shape[0] != target_tensor.shape[0]:
        pred_tensor = pred_tensor.repeat(target_tensor.shape[0], 1, 1, 1)

    batch_size = pred_tensor.shape[0]
    
    psnr_values = []
    ssim_values = []
    
    # Process each image in the batch individually
    for i in range(batch_size):
        single_metrics = calculate_metrics_single(pred_tensor[i], target_tensor[i])
        psnr_values.append(single_metrics['psnr'])
        ssim_values.append(single_metrics['ssim'])
    
    return {
        'psnr': sum(psnr_values) / len(psnr_values),
        'ssim': sum(ssim_values) / len(ssim_values)
    }