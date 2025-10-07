import torch
import kornia
import cv2
import os
import pandas as pd
from torch.utils.data import Dataset
from typing import Dict
from utils.coordinate_transforms import remap_image, create_polar_from_cartesian_maps # Assuming this function exists
import config

class DeblurDataset(Dataset):
    """Dataset for loading and preprocessing image pairs for deblurring."""
    def __init__(
        self,
        params_file: str,
        device: torch.device,
        img_height: int,
        img_width: int
    ):
        self.device = device
        self.dtype = torch.float32
        self.img_height = img_height
        self.img_width = img_width
        self.data_root = os.path.dirname(os.path.dirname(params_file))
        
        self.cart_to_polar_map_x, self.cart_to_polar_map_y = create_polar_from_cartesian_maps(
            self.img_height, self.img_width
        )

        self.df = pd.read_csv(params_file)
        print(f"Loaded {len(self.df)} entries from parameters file: {params_file}")
        print(f"Data root directory: {self.data_root}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """Returns a dictionary containing all processed image tensors for one sample."""
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        img_info = self.df.iloc[idx]
        
        blurry_path = os.path.join(self.data_root, img_info.blurred_path)
        sharp_path = os.path.join(self.data_root, img_info.original_path)

        # Load images
        blurry_img = cv2.imread(blurry_path)
        sharp_img = cv2.imread(sharp_path)

        # Add these lines to convert from BGR to RGB
        blurry_img = cv2.cvtColor(blurry_img, cv2.COLOR_BGR2RGB)
        sharp_img = cv2.cvtColor(sharp_img, cv2.COLOR_BGR2RGB)

        # Resize and convert to tensor
        blurry_img = cv2.resize(blurry_img, (self.img_width, self.img_height))
        sharp_img = cv2.resize(sharp_img, (self.img_width, self.img_height))
        
        blurry_cartesian = torch.from_numpy(blurry_img).permute(2, 0, 1).to(self.dtype) / 255.0
        sharp_cartesian = torch.from_numpy(sharp_img).permute(2, 0, 1).to(self.dtype) / 255.0

        # Move all tensors to the target device before the operation
        blurry_cartesian = blurry_cartesian.to(self.device)
        sharp_cartesian = sharp_cartesian.to(self.device)
        map_x_polar = self.cart_to_polar_map_x.to(self.device)
        map_y_polar = self.cart_to_polar_map_y.to(self.device)

        blurry_polar = remap_image(blurry_cartesian.unsqueeze(0), map_x_polar, map_y_polar).squeeze(0)
        sharp_polar = remap_image(sharp_cartesian.unsqueeze(0), map_x_polar, map_y_polar).squeeze(0)

        return {
            "blurry_polar": blurry_polar.to(self.device),
            "sharp_polar": sharp_polar.to(self.device),
            "blurry_cartesian": blurry_cartesian.to(self.device),
            "sharp_cartesian": sharp_cartesian.to(self.device),
            "blur_angle_deg": torch.tensor(img_info.blur_angle, dtype=self.dtype).to(self.device),
            "original_blur_angle": torch.tensor(img_info.original_blur_angle, dtype=self.dtype).to(self.device),
            "center_x_rel": torch.tensor(img_info.blur_center_x, dtype=self.dtype).to(self.device),
            "center_y_rel": torch.tensor(img_info.blur_center_y, dtype=self.dtype).to(self.device),
        }