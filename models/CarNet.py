import torch
import torch.nn as nn
import torch.fft
from utils.blur_operations import get_rotational_blur_psf # Assuming get_rotational_blur_psf is in this location

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1), nn.ReLU(True), nn.Conv2d(channels, channels, 3, 1, 1))
    def forward(self, x):
        return x + self.conv_block(x)

class AngleDetectionModule(nn.Module):
    def __init__(self, in_channels=3, features=64):
        super(AngleDetectionModule, self).__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, features, 4, 2, 1), nn.ReLU(True), ResBlock(features),
            nn.Conv2d(features, features*2, 4, 2, 1), nn.ReLU(True), ResBlock(features*2),
            nn.Conv2d(features*2, features*4, 4, 2, 1), nn.ReLU(True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        # The regressor now takes the concatenated image features and the noisy angle
        self.regressor = nn.Sequential(
            nn.Linear(features*4 + 1, features*2), 
            nn.ReLU(True), 
            nn.Linear(features*2, 1)
        )

    def forward(self, gp, noisy_angle_deg):
        # Extract image features
        img_features = self.backbone(gp)
        img_features = self.pool(img_features)
        img_features = self.flatten(img_features)

        # Reshape noisy_angle_deg to be 2D for concatenation
        if noisy_angle_deg.dim() == 1:
            noisy_angle_deg = noisy_angle_deg.unsqueeze(1)

        # Concatenate image features with the noisy angle
        combined_features = torch.cat([img_features, noisy_angle_deg], dim=1)

        # Predict the corrected angle
        corrected_angle = self.regressor(combined_features)
        return corrected_angle

class InversionModule(nn.Module):
    """
    Performs frequency-domain deconvolution using the provided kernel K.
    """
    def __init__(self, epsilon=1e-8):
        super(InversionModule, self).__init__()
        self.epsilon = epsilon

    def forward(self, gp, K_fft):
        """
        Args:
            gp (Tensor): The blurred image in the polar domain.
            K_fft (Tensor): The FFT of the blur kernel (PSF).
        """
        Gp_fft = torch.fft.fft2(gp, dim=(-2, -1))
        F_est_fft = Gp_fft / (K_fft + self.epsilon)
        f_est = torch.fft.ifft2(F_est_fft, dim=(-2, -1)).real
        return f_est

class RefinementStage(nn.Module):
    def __init__(self, num_stages=4, channels=3):
        super(RefinementStage, self).__init__()
        self.num_stages = num_stages
        self.residual_stages = nn.ModuleList([nn.Sequential(nn.Conv2d(channels * 2, 64, 3, 1, 1), ResBlock(64), nn.Conv2d(64, channels, 3, 1, 1)) for _ in range(num_stages)])
        self.final_refinement = nn.Sequential(nn.Conv2d(channels, 64, 3, 1, 1), ResBlock(64), nn.Conv2d(64, channels, 3, 1, 1))

    def forward(self, f_initial, gp):
        f_current = f_initial
        stage_outputs = [f_current]
        for stage in self.residual_stages:
            residual_input = torch.cat([f_current, gp], dim=1)
            residual = stage(residual_input)
            f_current = f_current + residual
            stage_outputs.append(f_current)
        f_final = self.final_refinement(f_current)
        stage_outputs.append(f_final)
        return stage_outputs

class CarNet(nn.Module):
    def __init__(self, num_stages=4, channels=3, n_blur_steps=30, 
                 use_angle_correction=True, use_refinement=True, return_intermediate=True):
        super(CarNet, self).__init__()
        self.n_blur_steps = n_blur_steps
        self.use_angle_correction = use_angle_correction
        self.use_refinement = use_refinement
        self.return_intermediate = return_intermediate
        
        self.initial_inversion = InversionModule()

        if self.use_angle_correction:
            self.angle_detector = AngleDetectionModule(in_channels=channels)
        
        if self.use_refinement:
            self.refinement_deblurrer = RefinementStage(num_stages=num_stages, channels=channels)

    def _generate_fft_kernel(self, image_tensor, angle_deg, center_x, center_y):
        """Helper function to generate a batch of FFT kernels from angles."""
        batch_size, c, h, w = image_tensor.shape
        psf_batch = []
        for i in range(batch_size):
            psf = get_rotational_blur_psf(
                image_shape=(c, h, w),
                center_x_relative=center_x[i],
                center_y_relative=center_y[i],
                angle_degrees=angle_deg[i],
                n_steps=self.n_blur_steps,
                device=image_tensor.device
            )
            psf_batch.append(psf)
        
        psf_tensor = torch.stack(psf_batch, dim=0)
        psf_shifted = torch.fft.ifftshift(psf_tensor, dim=(-2, -1))
        K_fft = torch.fft.fft2(psf_shifted, dim=(-2, -1))
        return K_fft

    def forward(self, gp, gt_angle_deg, center_x_rel, center_y_rel):
        K_fft_0 = self._generate_fft_kernel(gp, gt_angle_deg, center_x_rel, center_y_rel)
        f_stage0 = self.initial_inversion(gp, K_fft_0)

        # Initialize outputs
        all_stages = [f_stage0]
        
        if self.use_angle_correction:
            # Predict a residual angle from the initially deblurred image
            corrected_angle = self.angle_detector(f_stage0.detach(), gt_angle_deg)
        else:
            # Use original ground truth angle if angle correction is disabled
            corrected_angle = gt_angle_deg
            # Ensure residual_angle is a tensor, not None, for consistent output

        f_refined_initial = f_stage0
        # Perform a new inversion ONLY if the angle was corrected.
        if self.use_angle_correction:
            # Generate a new kernel with the corrected angle
            K_fft_1 = self._generate_fft_kernel(gp, corrected_angle, center_x_rel, center_y_rel)
            f_refined_initial = self.initial_inversion(gp, K_fft_1)

        # Stage 1: Refinement Deblur
        if self.use_refinement:
            # The refiner takes the (potentially new) initial deblur and the blurry image
            deblurred_stages = self.refinement_deblurrer(f_refined_initial, gp)
            
            # If angle was corrected, f_refined_initial is a new image.
            # We replace f_stage0 with the full sequence from the refiner.
            # Otherwise, the refiner started with f_stage0, so we just append the new stages.
            if self.use_angle_correction:
                all_stages = deblurred_stages
            else:
                all_stages.extend(deblurred_stages[1:])
        elif not self.use_refinement and self.use_angle_correction:
            # If not refining but using angle correction, we still need to account for the refined output
            all_stages.append(f_refined_initial)

        # Final logic for what to return
        if not self.return_intermediate:
            # Return only the final stage from the list
            all_stages = [all_stages[-1]]
        
        return all_stages, corrected_angle