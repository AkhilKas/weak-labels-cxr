# weak-labels-cxr
Exploring how to handle uncertain labels in medical imaging — experiments on CheXpert for robust chest X-ray classification

# Data Setup

This project uses the CheXpert-v1.0-small dataset.

## Download Instructions

1. Go to [Kaggle CheXpert Dataset](https://www.kaggle.com/datasets/ashery/chexpert)
2. Download `archive.zip` (~11GB)
3. Place it in this `data/` folder
4. Unzip it:
```bash
   cd data/
   unzip archive.zip
```
5. Run preprocessing:
```bash
   python scripts/prepare_data.py
```

## Expected Structure After Setup
```
data/
├── archive/
│   ├── train/
│   ├── valid/
│   ├── train.csv
│   └── valid.csv
└── processed/
    ├── train_labels.csv
    └── valid_labels.csv
README.md
```