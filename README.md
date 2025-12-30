# Two-stage PE Recommender

A two-stage recommender system for matching investors with investment opportunities using Two-Tower retrieval and Deep Ranking models.

## Overview

This project implements a sophisticated recommendation system that:
- Uses a Two-Tower model for fast candidate retrieval
- Applies a Deep Ranking model with pairwise learning for precise ranking
- Handles implicit feedback data
- Includes comprehensive evaluation metrics
- If you see discrepancy between the input dataset format and various classes design, feel free to reach out to the author to get practical advice for smooth implementation. 

## Project Structure

```
PrivateEquity_RecSys/
│
├── data/                           # Data directory (created by scripts)
│   ├── enhanced_interactions.csv
│   ├── investor_features.csv
│   └── deal_features.csv
│
├── models/                         # Model artifacts (created by scripts)
│   ├── checkpoints/               # Model checkpoints
│   ├── feature_encoder.pkl        # Feature encoder
│   ├── training_summary.json      # Training metadata
│   └── evaluation_results.json    # Evaluation results
│
├── notebooks/
│   ├── data_generation.ipynb      # Generate synthetic data
│   ├── model_training.ipynb       # Train both models
│   └── model_evaluation.ipynb     # Evaluate performance
│
├── models.py                       # Model architectures
├── datasets.py                     # Dataset classes
├── evaluation.py                   # Evaluation utilities
└── README.md                       # This file
```

## Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

Create a `requirements.txt` file with:
```
torch>=1.9.0
pytorch-lightning>=1.5.0
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
tqdm>=4.62.0
```

## Usage

### 1. Generate Data

Run the data generation notebook to create synthetic investor-deal interaction data:

```bash
jupyter notebook notebooks/data_generation.ipynb
```

This creates:
- 125 investors with features (type, region, risk profile, investment range)
- 1,000 deals with features (sector, stage, region, financials)
- 2,000 interactions with preference-based patterns

### 2. Train Models

Train both the Two-Tower and Deep Ranking models:

```bash
jupyter notebook notebooks/model_training.ipynb
```

This trains:
- **Two-Tower Model**: For fast similarity-based retrieval
- **Deep Ranking Model**: For precise pairwise ranking

### 3. Evaluate Performance

Run comprehensive evaluation:

```bash
jupyter notebook notebooks/model_evaluation.ipynb
```

This computes:
- Hit@10, NDCG@10, MRR, Recall@10, Precision@10
- Single model performance
- Two-stage pipeline performance
- Performance by investor segments

## Model Architecture

### Two-Tower Model
- Separate neural networks for investors and deals
- L2-normalized embeddings
- Temperature-scaled dot product similarity
- Optimized for retrieval efficiency

### Deep Ranking Model  
- Concatenated investor-deal features
- Deep MLP with BatchNorm and Dropout
- RankNet pairwise loss
- Optimized for ranking quality

## Key Features

- **Implicit Feedback**: Handles binary interaction data
- **Negative Sampling**: Intelligent sampling of non-interactions
- **Pairwise Learning**: Direct optimization for ranking
- **Two-Stage Pipeline**: Combines efficiency and accuracy
- **Comprehensive Evaluation**: Multiple ranking metrics

## Extending the System

### Adding New Features
1. Update feature generation in `data_generation.ipynb`
2. Modify `FeatureEncoder` in `models.py`
3. Update embedding dimensions in model architectures

### Using Real Data
1. Replace synthetic data generation with data loading
2. Ensure proper feature encoding
3. Adjust model dimensions accordingly

### Production Deployment
1. Use the trained models from checkpoints
2. Implement online serving with batch scoring
3. Add feature preprocessing pipeline

## Citation

If you use this code, please cite:
```
@software{PrivateEquity_RecSys,
  title = {Two-stage PE Recommender},
  author = {czhang165},
  year = {2025},
  url = {https://github.com/czhang165/PrivateEquity_RecSys}
}
```

## License

GNU General Public License v3.0 - see LICENSE file for details
