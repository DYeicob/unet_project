import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SegmentationDataset(Dataset):
    """
    Custom Dataset for image segmentation tasks.
    Loads images and their corresponding masks, applying consistent 
    spatial augmentations to both.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = sorted(os.listdir(image_dir))
        
        # Default to basic tensor conversion if no transform is provided
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Construct file paths
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])

        # Load image using OpenCV and convert BGR to RGB
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Load mask in grayscale mode
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Manual spatial augmentations
        # We apply them manually here to ensure the exact same transformation 
        # is applied to both the image and the mask.
        
        # Random horizontal flip (50% chance)
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            
        # Random vertical flip (50% chance)
        if np.random.rand() > 0.5:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)

        # Image Preprocessing: Normalize to [0, 1] and convert to Torch Tensor
        # Transpose from HWC (OpenCV default) to CHW (PyTorch default)
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image.transpose(2, 0, 1))

        # Mask Preprocessing: Binarize and add channel dimension
        mask = mask.astype(np.float32) / 255.0
        mask = (mask > 0.5).astype(np.float32) # Convert to binary 0/1
        mask = torch.from_numpy(mask).unsqueeze(0) # HW to 1HW

        return image, mask
