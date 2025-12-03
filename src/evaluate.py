"""
Evaluation metrics for CheXpert.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.dataset import LABELS


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Evaluate model on a dataset.
    
    Returns:
        Dictionary with AUC per label and mean AUC.
    """
    model.eval()
    
    all_logits = []
    all_labels = []
    
    for batch in tqdm(loader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        labels = batch["labels"]
        
        logits = model(images)
        
        all_logits.append(logits.cpu())
        all_labels.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    
    # Convert logits to probabilities
    probs = 1 / (1 + np.exp(-all_logits))
    
    # Compute AUC for each label
    auc_scores = {}
    for i, label in enumerate(LABELS):
        y_true = all_labels[:, i]
        y_pred = probs[:, i]
        
        # Skip if only one class present
        if len(np.unique(y_true)) < 2:
            auc_scores[label] = 0.5
        else:
            auc_scores[label] = roc_auc_score(y_true, y_pred)
    
    mean_auc = np.mean(list(auc_scores.values()))
    
    return {
        "auc": auc_scores,
        "mean_auc": mean_auc,
    }