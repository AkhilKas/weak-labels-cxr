"""
Model architecture for CheXpert classification.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class CheXpertModel(nn.Module):
    """
    Multi-label classifier for CheXpert.
    
    Args:
        backbone: Name of timm backbone (e.g., 'efficientnet_b0', 'densenet121')
        num_classes: Number of output classes
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate before classifier
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
        )
        
        encoder_dim = self.encoder.num_features
        
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(encoder_dim, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)


def get_loss_fn(strategy: str = "zeros", pos_weight: Optional[torch.Tensor] = None):
    """
    Get loss function based on uncertainty strategy.
    
    Args:
        strategy: Label strategy ('zeros', 'ones', 'ignore', 'soft')
        pos_weight: Optional positive class weights for imbalanced data
    """
    if strategy in ["zeros", "ones", "soft"]:
        return nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    elif strategy == "ignore":
        # Custom loss that ignores -1 labels
        def ignore_uncertain_loss(logits, targets):
            mask = targets != -1
            if mask.sum() == 0:
                return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[mask],
                targets[mask],
                reduction="mean",
            )
            return loss
        
        return ignore_uncertain_loss
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


# Quick test
if __name__ == "__main__":
    model = CheXpertModel(backbone="efficientnet_b0", pretrained=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Backbone: efficientnet_b0")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")