"""
CheXpert Dataset for PyTorch.
"""

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, Callable


LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


class CheXpertDataset(Dataset):
    """
    CheXpert Dataset.
    
    Args:
        csv_path: Path to processed CSV file
        transform: Optional image transforms
        frontal_only: If True, only use frontal views (recommended)
    """
    
    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        frontal_only: bool = True,
    ):
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        
        if frontal_only:
            self.df = self.df[self.df["Frontal/Lateral"] == "Frontal"].reset_index(drop=True)
        
        # Fill missing labels with 0 (treat as negative)
        self.df[LABELS] = self.df[LABELS].fillna(0)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        
        # Load image
        image = Image.open(row["Path"]).convert("RGB")
        
        if self.transform:
            image = self.transform(image)
        
        # Get labels
        labels = torch.tensor(row[LABELS].values.astype(float), dtype=torch.float32)
        
        return {
            "image": image,
            "labels": labels,
            "path": row["Path"],
        }


def get_transforms(image_size: int = 224, train: bool = True):
    """
    Get image transforms.
    
    Args:
        image_size: Target image size
        train: If True, apply training augmentations
    """
    from torchvision import transforms
    
    if train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])


# Quick test
if __name__ == "__main__":
    dataset = CheXpertDataset(
        csv_path="data/processed/train_zeros.csv",
        transform=get_transforms(train=True),
        frontal_only=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Image shape: {sample['image'].shape}")
    print(f"Labels: {sample['labels']}")
    print(f"Path: {sample['path']}")