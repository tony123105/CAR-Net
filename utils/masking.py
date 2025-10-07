import torch
from typing import Tuple

def apply_circular_mask(tensor: torch.Tensor, mask_color: float = 0.5) -> torch.Tensor:
    """
    Applies a circular mask to the tensor, keeping only the central circular region.
    
    Args:
        tensor: Input tensor of shape (C, H, W)
        mask_color: Color value for the masked area (0.5 for gray)
    
    Returns:
        Tensor with circular mask applied
    """
    _, H, W = tensor.shape
    center_x, center_y = W // 2, H // 2
    radius = min(H, W) // 2
    
    # Create coordinate grids
    y, x = torch.meshgrid(torch.arange(H, device=tensor.device), torch.arange(W, device=tensor.device), indexing='ij')
    
    # Calculate distance from center
    distance = torch.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    
    # Create circular mask (True for points inside the circle)
    circular_mask = distance <= radius
    
    # Apply mask
    masked_tensor = tensor.clone()
    for c in range(tensor.shape[0]):
        masked_tensor[c][~circular_mask] = mask_color
    
    return masked_tensor

def apply_gray_border(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies a neutral gray border to both prediction and target tensors.
    """

    # A small epsilon is used to handle potential floating point inaccuracies.
    border_mask = torch.sum(pred, dim=1, keepdim=True) < 1e-4
    
    # Clone tensors to avoid modifying originals
    pred_bordered = pred.clone()
    target_bordered = target.clone()

    # Apply gray border (0.5 is gray in a [0, 1] normalized space)
    pred_bordered[border_mask.expand_as(pred)] = 0.5
    target_bordered[border_mask.expand_as(target)] = 0.5
    
    # Apply circular mask to focus on the central region only
    pred_bordered = apply_circular_mask(pred_bordered[0]).unsqueeze(0) if pred_bordered.dim() == 4 else apply_circular_mask(pred_bordered)
    target_bordered = apply_circular_mask(target_bordered[0]).unsqueeze(0) if target_bordered.dim() == 4 else apply_circular_mask(target_bordered)
    
    return pred_bordered, target_bordered