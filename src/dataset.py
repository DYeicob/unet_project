import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        
        # Basic transformations if none provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # Load image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Apply simple augmentations manually to ensure consistency between image and mask
        # Note: For more complex augmentations, libraries like albumentations are recommended.
        # Here we implement simple flip/rotate as requested.
        
        # Random horizontal flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            
        # Random vertical flip
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Normalize image to [0, 1] and convert to tensor
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1)) # HWC to CHW

        # Prepare mask: 1 channel, values 0 or 1
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32) # Binarize
        mask = torch.from_numpy(mask).unsqueeze(0) # Add channel dim: HW -> 1HW

        return image, mask
