"""
Preprocess CheXpert dataset.

Fixes path mismatch and creates label files with different uncertainty strategies.
"""

import pandas as pd
from pathlib import Path

# Paths
RAW_DIR = Path("data/archive")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(exist_ok=True)

# Competition labels (5 tasks with most uncertainty)
LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]


def fix_image_path(path: str) -> str:
    """Convert CSV path to actual path in our folder structure."""
    # CheXpert-v1.0-small/train/patient... -> data/archive/train/patient...
    return path.replace("CheXpert-v1.0-small/", "data/archive/")


def load_and_clean(csv_path: Path) -> pd.DataFrame:
    """Load CSV and fix paths."""
    df = pd.read_csv(csv_path)
    df["Path"] = df["Path"].apply(fix_image_path)
    return df


def apply_uncertainty_strategy(df: pd.DataFrame, strategy: str) -> pd.DataFrame:
    """
    Handle uncertain labels (-1) with different strategies.
    
    Strategies:
        - zeros: Map -1 -> 0
        - ones: Map -1 -> 1
        - ignore: Keep -1 (filter during training)
        - soft: Map -1 -> 0.5
    """
    df = df.copy()
    
    for label in LABELS:
        if strategy == "zeros":
            df[label] = df[label].replace(-1, 0)
        elif strategy == "ones":
            df[label] = df[label].replace(-1, 1)
        elif strategy == "soft":
            df[label] = df[label].replace(-1, 0.5)
        elif strategy == "ignore":
            pass  # Keep -1 as is
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df


def compute_label_stats(df: pd.DataFrame, name: str) -> None:
    """Print label distribution statistics."""
    print(f"\n{name} Label Distribution:")
    print("-" * 50)
    
    for label in LABELS:
        total = len(df)
        positive = (df[label] == 1).sum()
        negative = (df[label] == 0).sum()
        uncertain = (df[label] == -1).sum()
        missing = df[label].isna().sum()
        
        print(f"{label}:")
        print(f"  Positive: {positive:>6} ({100*positive/total:.1f}%)")
        print(f"  Negative: {negative:>6} ({100*negative/total:.1f}%)")
        print(f"  Uncertain: {uncertain:>6} ({100*uncertain/total:.1f}%)")
        print(f"  Missing: {missing:>6} ({100*missing/total:.1f}%)")


def main():
    print("Loading raw data...")
    train_df = load_and_clean(RAW_DIR / "train.csv")
    valid_df = load_and_clean(RAW_DIR / "valid.csv")
    
    print(f"Train samples: {len(train_df)}")
    print(f"Valid samples: {len(valid_df)}")
    
    # Show original label stats
    compute_label_stats(train_df, "Train (Original)")
    
    # Keep only relevant columns
    keep_cols = ["Path", "Sex", "Age", "Frontal/Lateral"] + LABELS
    train_df = train_df[keep_cols]
    valid_df = valid_df[keep_cols]
    
    # Save with different uncertainty strategies
    strategies = ["zeros", "ones", "ignore", "soft"]
    
    for strategy in strategies:
        train_out = apply_uncertainty_strategy(train_df, strategy)
        valid_out = apply_uncertainty_strategy(valid_df, strategy)
        
        train_out.to_csv(PROCESSED_DIR / f"train_{strategy}.csv", index=False)
        valid_out.to_csv(PROCESSED_DIR / f"valid_{strategy}.csv", index=False)
        
        print(f"\nSaved: train_{strategy}.csv, valid_{strategy}.csv")
    
    print("\nDone! Processed files saved to data/processed/")


if __name__ == "__main__":
    main()
