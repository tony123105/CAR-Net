import torch
import torch.nn.functional as F
import kornia
from typing import Tuple

def simple_rotational_blur(
    sharp_images: torch.Tensor,
    center_x_relative: torch.Tensor,
    center_y_relative: torch.Tensor,
    angle_degrees: torch.Tensor,
    n_steps: int = 15
) -> torch.Tensor:
    """
    A basic, efficient rotational blur using backward mapping with a fixed
    number of samples. May produce artifacts (central pinprick, aliasing).
    """
    device = sharp_images.device
    batch_size, channels, height, width = sharp_images.shape

    # Prepare parameters
    center_x = center_x_relative.squeeze() * width
    center_y = center_y_relative.squeeze() * height
    angle_rad = angle_degrees.squeeze() * (torch.pi / 180.0)

    # Create grid and calculate polar coordinates
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32, device=device),
        torch.arange(width, dtype=torch.float32, device=device),
        indexing='ij'
    )
    rel_x = xx - center_x
    rel_y = yy - center_y
    radius = torch.sqrt(rel_x**2 + rel_y**2)
    phi_base = torch.atan2(rel_y, rel_x)

    # Create fixed angle steps for the blur
    angle_steps = torch.linspace(-angle_rad / 2, angle_rad / 2, n_steps, device=device)
    phi_arc = phi_base.unsqueeze(0) + angle_steps.view(-1, 1, 1)

    # Calculate sample points
    sample_x = center_x + radius.unsqueeze(0) * torch.cos(phi_arc)
    sample_y = center_y + radius.unsqueeze(0) * torch.sin(phi_arc)

    # Normalize coordinates and sample from the sharp image
    sample_x_norm = (sample_x / (width - 1)) * 2 - 1
    sample_y_norm = (sample_y / (height - 1)) * 2 - 1
    grid = torch.stack((sample_x_norm, sample_y_norm), dim=-1)
    
    sharp_images_expanded = sharp_images.repeat_interleave(n_steps, dim=0)
    batch_grid_flat = grid.view(batch_size * n_steps, height, width, 2)
    sampled_pixels_flat = F.grid_sample(
        sharp_images_expanded, batch_grid_flat, mode='bilinear', padding_mode='border', align_corners=True
    )

    # Reshape and average over the samples
    sampled_pixels = sampled_pixels_flat.view(batch_size, n_steps, channels, height, width)
    blurred_image = torch.mean(sampled_pixels, dim=1)

    return blurred_image

def get_rotational_blur_psf(
    image_shape: tuple,
    center_x_relative: torch.Tensor,
    center_y_relative: torch.Tensor,
    angle_degrees: torch.Tensor,
    n_steps: int,
    device: torch.device
) -> torch.Tensor:
    """
    Generates the Point Spread Function (PSF) for the rotational blur.

    The PSF is obtained by applying the blur operation to an impulse image
    (an image with a single bright pixel at the center).

    Args:
        image_shape (tuple): The shape of the output PSF (C, H, W).
        center_x_relative (torch.Tensor): Relative center x-coordinate for the blur.
        center_y_relative (torch.Tensor): Relative center y-coordinate for the blur.
        angle_degrees (torch.Tensor): The total angle of rotation.
        n_steps (int): The number of steps to approximate the blur.
        device (torch.device): The device to perform computation on.

    Returns:
        torch.Tensor: The spatial blur kernel (PSF) as a tensor.
    """
    c, h, w = image_shape
    
    # Create a batch of one impulse image (a single white pixel at the center)
    impulse = torch.zeros(1, c, h, w, device=device)
    impulse[:, :, h // 2, w // 2] = 1.0

    # Apply the existing blur function to the impulse image
    # The result is the Point Spread Function (PSF), i.e., the kernel.
    psf = simple_rotational_blur(
        sharp_images=impulse,
        center_x_relative=center_x_relative.view(1),
        center_y_relative=center_y_relative.view(1),
        angle_degrees=angle_degrees.view(1),
        n_steps=n_steps
    )

    # Return the resulting PSF (remove the batch dimension)
    return psf.squeeze(0)