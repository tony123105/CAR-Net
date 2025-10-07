import torch
import torch.nn as nn
import kornia
from typing import List, Tuple, Dict
from utils.blur_operations import simple_rotational_blur

class PhysicsLoss(nn.Module):
    """
    Calculates the physics-based loss by re-blurring each stage's output
    and comparing it to the original blurry input.
    """
    def __init__(self, n_steps: int):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        self.n_steps = n_steps

    def forward(self, **kwargs):
        """
        Calculates the physics-based loss.

        Args:
            stage_outputs_cartesian (List[torch.Tensor]): The deblurred images predicted by the model at each stage.
            gt_blur (torch.Tensor): The original ground truth blurred image.
            blur_params (dict): A dictionary containing the blur parameters for each image in the batch.
                                Expected keys: 'center_x_relative', 'center_y_relative', 'angle_degrees'.

        Returns:
            torch.Tensor: The calculated physics loss.
        """
        stage_outputs = kwargs['stage_outputs_cartesian']
        gt_blur_cartesian = kwargs['blurry_cartesian']
        blur_angles = kwargs['estimated_angle'] 

        total_loss = 0.0
        for pred_unfolded_sharp in stage_outputs:
            reblurred_list = []
            # Iterate over each item in the batch
            for i in range(pred_unfolded_sharp.shape[0]):
                # Apply blur to a single predicted sharp image using its corresponding parameters
                # Force the center of rotation to be the middle of the image.
                center_x = torch.tensor(0.5, device=pred_unfolded_sharp.device)
                center_y = torch.tensor(0.5, device=pred_unfolded_sharp.device)
                
                reblurred_image = simple_rotational_blur(
                    sharp_images=pred_unfolded_sharp[i].unsqueeze(0),  # Add batch dim
                    center_x_relative=center_x,
                    center_y_relative=center_y,
                    angle_degrees=blur_angles[i],
                    n_steps=self.n_steps
                )
                reblurred_list.append(reblurred_image)

            # Stack the re-blurred images back into a single batch tensor
            reblurred_batch = torch.cat(reblurred_list, dim=0)

            # Calculate the L1 loss between the re-blurred prediction and the original blurred image
            total_loss += self.l1_loss(reblurred_batch, gt_blur_cartesian)
        
        return total_loss / len(stage_outputs)
    

class L1ReconstructionLoss(nn.Module):
    """
    Calculates the L1 (MAE) loss between each deblurred stage and the sharp ground truth.
    """
    def __init__(self):
        super().__init__()
        self.base_loss = nn.L1Loss()

    def forward(self, stage_outputs_cartesian: List[torch.Tensor], sharp_cartesian: torch.Tensor, **kwargs) -> torch.Tensor:
        total_loss = 0.0
        for output_cartesian in stage_outputs_cartesian:
            if output_cartesian.shape[0] != sharp_cartesian.shape[0]:
                output_cartesian = output_cartesian.repeat(sharp_cartesian.shape[0], 1, 1, 1)
            total_loss += self.base_loss(output_cartesian, sharp_cartesian)
        return total_loss / len(stage_outputs_cartesian)

class SSIMLoss(nn.Module):
    """
    Calculates the SSIM loss between each deblurred stage and the sharp ground truth.
    """
    def __init__(self, window_size: int = 11):
        super().__init__()
        # Kornia's SSIM loss returns 1-SSIM, lower is better
        self.base_loss = kornia.losses.SSIMLoss(window_size=window_size)

    def forward(self, stage_outputs_cartesian: List[torch.Tensor], sharp_cartesian: torch.Tensor, **kwargs) -> torch.Tensor:
        total_loss = 0.0
        for output_cartesian in stage_outputs_cartesian:
            if output_cartesian.shape[0] != sharp_cartesian.shape[0]:
                output_cartesian = output_cartesian.repeat(sharp_cartesian.shape[0], 1, 1, 1)
            total_loss += self.base_loss(output_cartesian, sharp_cartesian)
        return total_loss / len(stage_outputs_cartesian)

class CombinedLoss(nn.Module):
    """
    A weighted combination of multiple loss functions.
    Handles different model outputs gracefully.
    """
    def __init__(self, physics_weight=0.5, recon_weight=0.5, ssim_weight=0.0, angle_weight=0.1, n_steps_physics=15, max_angle=40.0):
        super().__init__()
        self.physics_weight = physics_weight
        self.recon_weight = recon_weight
        self.ssim_weight = ssim_weight
        self.angle_weight = angle_weight
        self.max_angle = max_angle # Store max angle for normalization

        self.physics_loss = PhysicsLoss(n_steps=n_steps_physics) if physics_weight > 0 else None
        self.recon_loss = L1ReconstructionLoss() if recon_weight > 0 else None
        self.ssim_loss = SSIMLoss() if ssim_weight > 0 else None
        self.angle_loss = nn.L1Loss() if angle_weight > 0 else None

    def forward(self, **kwargs) -> Tuple[torch.Tensor, Dict[str, float]]:
        total_loss = 0
        loss_dict = {}

        if self.physics_loss and self.physics_weight > 0:
            loss = self.physics_weight * self.physics_loss(**kwargs)
            total_loss += loss
            loss_dict["physics_loss"] = loss.item()

        if self.recon_loss and self.recon_weight > 0:
            loss = self.recon_weight * self.recon_loss(**kwargs)
            total_loss += loss
            loss_dict["recon_loss"] = loss.item()

        if self.ssim_loss and self.ssim_weight > 0:
            loss = self.ssim_weight * self.ssim_loss(**kwargs)
            total_loss += loss
            loss_dict["ssim_loss"] = loss.item()

        if self.angle_loss and self.angle_weight > 0 and 'estimated_angle' in kwargs and 'blur_angle_deg' in kwargs:
            estimated = kwargs['estimated_angle']
            target = kwargs['blur_angle_deg'].to(estimated.device).view_as(estimated)

            # Normalize both estimated and target angles to the [0, 1] range
            norm_estimated = estimated / self.max_angle
            norm_target = target / self.max_angle
            
            loss = self.angle_weight * self.angle_loss(norm_estimated, norm_target)
            total_loss += loss
            loss_dict["angle_loss"] = loss.item()

        loss_dict["total_loss"] = total_loss.item()
        return total_loss, loss_dict