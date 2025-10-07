import torch
import torch.nn.functional as F
import math
from typing import Tuple

def create_polar_from_cartesian_maps(height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates mapping grids to convert an image from Cartesian to Polar coordinates.
    The output maps specify, for each pixel in the destination polar image,
    which coordinate to sample from in the source cartesian image.

    Args:
        height (int): Image height.
        width (int): Image width.

    Returns:
        A tuple of two tensors (map_x, map_y), each with shape (1, H, W).
    """
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0
    
    # Create a grid of polar coordinates for the destination image
    rho = torch.linspace(0, min(center_x, center_y), height)
    phi = torch.linspace(-math.pi, math.pi, width) # Use -pi to pi for atan2 consistency
    rho_grid, phi_grid = torch.meshgrid(rho, phi, indexing='ij')

    # Convert these polar coordinates back to cartesian coordinates to find the source pixel
    map_x = center_x + rho_grid * torch.cos(phi_grid)
    map_y = center_y + rho_grid * torch.sin(phi_grid)
    
    # Add batch dimension
    return map_x.unsqueeze(0), map_y.unsqueeze(0)

def create_cartesian_from_polar_maps(height: int, width: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Creates mapping grids to convert an image from Polar to Cartesian coordinates.
    The output maps specify, for each pixel in the destination cartesian image,
    which coordinate to sample from in the source polar image.

    Args:
        height (int): Image height.
        width (int): Image width.

    Returns:
        A tuple of two tensors (map_x, map_y), each with shape (1, H, W).
    """
    center_x, center_y = (width - 1) / 2.0, (height - 1) / 2.0

    # Create a grid of cartesian coordinates for the destination image
    y_grid, x_grid = torch.meshgrid(torch.arange(float(height)), torch.arange(float(width)), indexing='ij')
    
    # Translate coordinates to be relative to the center
    x_centered = x_grid - center_x
    y_centered = y_grid - center_y
    
    # Convert cartesian to polar coordinates to find the source pixel
    rho = torch.sqrt(x_centered**2 + y_centered**2)
    phi = torch.atan2(y_centered, x_centered)

    # map_x corresponds to the 'phi' dimension (width)
    map_x = (phi + math.pi) / (2 * math.pi) * (width - 1)
    # map_y corresponds to the 'rho' dimension (height)
    map_y = rho / min(center_x, center_y) * (height - 1)

    # Add batch dimension
    return map_x.unsqueeze(0), map_y.unsqueeze(0)

def remap_image(
    image: torch.Tensor, 
    map_x: torch.Tensor, 
    map_y: torch.Tensor
) -> torch.Tensor:
    """
    Remaps an image using coordinate maps with proper normalization for grid_sample.
    
    Args:
        image (torch.Tensor): The input image tensor (B, C, H, W).
        map_x (torch.Tensor): The x-coordinate map in pixel units (B, H, W).
        map_y (torch.Tensor): The y-coordinate map in pixel units (B, H, W).

    Returns:
        torch.Tensor: The remapped image.
    """
    batch_size, _, height, width = image.shape
    
    # grid_sample requires coordinates in the range [-1, 1].
    normalized_map_x = 2.0 * map_x / (width - 1) - 1.0
    normalized_map_y = 2.0 * map_y / (height - 1) - 1.0
    
    # Create the grid tensor of shape (B, H, W, 2)
    grid = torch.stack((normalized_map_x, normalized_map_y), dim=-1)
    
    # Perform the remapping
    remapped = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return remapped