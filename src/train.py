"""
Training script for CheXpert.
"""

import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import wandb

from src.dataset import CheXpertDataset, get_transforms, LABELS
from src.model import CheXpertModel, get_loss_fn
from src.utils import set_seed, get_device, AverageMeter, EarlyStopping
from src.evaluate import evaluate


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    epoch: int,
) -> float:
    model.train()
    losses = AverageMeter()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        images = batch["image"].to(device)
        labels = batch["labels"].to(device)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), images.size(0))
        pbar.set_postfix({"loss": f"{losses.avg:.4f}"})
    
    return losses.avg


def train(config: dict):
    # Setup
    set_seed(config["seed"])
    device = get_device()
    print(f"Using device: {device}")
    
    # Output directory
    output_dir = Path("outputs") / config["experiment_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Data
    strategy = config["labels"]["uncertainty_strategy"]
    train_csv = f"data/processed/train_{strategy}.csv"
    valid_csv = f"data/processed/valid_{strategy}.csv"
    
    train_dataset = CheXpertDataset(
        csv_path=train_csv,
        transform=get_transforms(config["data"]["image_size"], train=True),
        frontal_only=True,
    )
    valid_dataset = CheXpertDataset(
        csv_path=valid_csv,
        transform=get_transforms(config["data"]["image_size"], train=False),
        frontal_only=True,
    )
    
    pin_memory = device.type == "cuda"
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["data"]["num_workers"],
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["data"]["num_workers"],
        pin_memory=pin_memory,
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    
    # Model
    model = CheXpertModel(
        backbone=config["model"]["backbone"],
        num_classes=len(LABELS),
        pretrained=config["model"]["pretrained"],
        dropout=config["model"]["dropout"],
    ).to(device)
    
    # Loss and optimizer
    loss_fn = get_loss_fn(strategy)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["epochs"],
    )
    
    # Early stopping
    early_stopping = EarlyStopping(
        patience=config["training"].get("patience", 3),
        mode="max",
    )
    
    # Wandb
    if config["logging"]["use_wandb"]:
        wandb.init(
            project=config["logging"]["project"],
            name=config["experiment_name"],
            config=config,
        )
    
    # Training loop
    best_auc = 0.0
    
    for epoch in range(1, config["training"]["epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, device, epoch
        )
        
        metrics = evaluate(model, valid_loader, device)
        scheduler.step()
        
        # Logging
        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "lr": scheduler.get_last_lr()[0],
            "mean_auc": metrics["mean_auc"],
        }
        for label in LABELS:
            log_dict[f"auc_{label}"] = metrics["auc"][label]
        
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, mean_auc={metrics['mean_auc']:.4f}")
        
        if config["logging"]["use_wandb"]:
            wandb.log(log_dict)
        
        # Save best model
        if metrics["mean_auc"] > best_auc:
            best_auc = metrics["mean_auc"]
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "metrics": metrics},
                output_dir / "best_model.pth",
            )
            print(f"  New best model saved (AUC: {best_auc:.4f})")
        
        # Early stopping check
        if early_stopping(metrics["mean_auc"]):
            print(f"Early stopping triggered at epoch {epoch}")
            break
    
    # Save final model
    torch.save(
        {"epoch": epoch, "model": model.state_dict(), "metrics": metrics},
        output_dir / "final_model.pth",
    )
    
    if config["logging"]["use_wandb"]:
        wandb.finish()
    
    print(f"\nTraining complete. Best AUC: {best_auc:.4f}")
    return best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    train(config)