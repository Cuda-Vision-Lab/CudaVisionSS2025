# Assignment 6: Siamese Networks and Contrastive Learning

## 📋 Overview

This assignment focuses on implementing and comparing two self-supervised learning approaches for face recognition: **TriNet Siamese Networks** and **SimCLR (Simple Framework for Contrastive Learning)**. The project uses the Labeled Faces in the Wild (LFW) dataset to learn discriminative face embeddings without explicit class labels during training.

<div align="center">
  <figure>
    <img src="imgs/Screenshot%202025-12-29%20at%2021.34.30.png" alt="Anchor, positive, and negative images" width="70%" height="80%">
    <figcaption><b>Caption:</b> Example triplet input for Siamese network: anchor, positive (same identity), and negative (different identity) images.</figcaption>
  </figure>
</div>


## 🎯 Objectives

- Implement TriNet Siamese model with triplet loss for face recognition
- Implement SimCLR model with contrastive learning for face embeddings
- Use ResNet-18 as convolutional backbone with custom projection heads
- Train and evaluate both models on the LFW dataset
- Find adequate temperature and margin hyperparameters
- Compare model performance and embedding quality
- Visualize embeddings using PCA and t-SNE
- **Extra Credit**: Test models on group member photos and find celebrity look-alikes

## 📊 Dataset

**Labeled Faces in the Wild (LFW)**
- **Source**: [Kaggle Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
- **Total images**: Over 13,000 face images
- **People/Classes**: 34 people (with minimum 50 faces per person)
- **Image size**: 154×154×3 (RGB) - cropped to face region
- **Train/Test split**: 80/20 (stratified)
- **Preprocessing**: Images resized to 64×64 for model input

The dataset is loaded using `sklearn.datasets.fetch_lfw_people` with:
- `min_faces_per_person=50`: Filter to people with at least 50 images
- `color=True`: Load RGB images
- `resize=1.0`: Original resolution
- `slice_=(slice(48, 202), slice(48, 202))`: Crop to face region

## 🏗️ Models Implemented

### 1. TriNet Siamese Model

A Siamese network architecture that learns embeddings by minimizing distance between similar faces and maximizing distance between different faces.

**Architecture**:
- **Backbone**: ResNet-18 (up to AvgPool layer)
  - Modified first conv layer: `Conv2d(3, 64, kernel_size=7, stride=2, padding=3)`
  - Removed final FC layer (replaced with `Identity`)
  - Output: 512-dimensional features
- **Projection Head**: 
  - `Linear(512 → 64)`
  - `ReLU`
  - `Linear(64 → 64)`
- **Normalization**: L2 normalization layer
- **Embedding dimension**: 64

**Training**:
- **Loss**: Triplet Loss with temperature scaling
- **Loss function**: `max(0, d(anchor, positive) - d(anchor, negative) + margin)`
- **Temperature scaling**: Embeddings scaled by temperature before distance computation
- **Optimizer**: Adam (lr=1e-4, weight_decay=1e-5)
- **Training iterations**: 10,000 iterations (configurable)
- **Batch size**: 64
- **Validation**: Every 250 iterations

**Key Features**:
- Efficient forward pass: Processes anchor, positive, and negative in a single batch
- Temperature-scaled distances for better gradient flow
- Margin parameter controls separation between positive and negative pairs

<div align="center">
  <img src="imgs/Screenshot 2025-12-29 at 21.34.00.png" alt="TriNet Training Visualization" width="70%" height="80%">
  <br>
  <em>Sample output from TriNet Siamese face representation learning</em>
</div>


### 2. SimCLR Model

A contrastive learning framework that learns representations by maximizing agreement between differently augmented views of the same image.

**Architecture**:
- **Backbone**: ResNet-18 (up to AvgPool layer)
  - Modified first conv layer: `Conv2d(3, 64, kernel_size=7, stride=2, padding=3)`
  - Removed final FC layer (replaced with `Identity`)
  - Output: 512-dimensional features
- **Projection Head**:
  - `Linear(512 → 512)`
  - `ReLU`
  - `Linear(512 → 128)`
- **Normalization**: L2 normalization
- **Embedding dimension**: 128

**Training**:
- **Loss**: Normalized Temperature-scaled Cross Entropy (NT-Xent) Loss
- **Loss function**: Contrastive loss over similarity matrix of augmented pairs
- **Optimizer**: Adam (lr=3e-3)
- **Epochs**: 300
- **Batch size**: 64
- **Data augmentation**: 
  - Random resized crop (scale: 0.5-1.33)
  - Random rotation (±20 degrees)
  - Random horizontal flip (p=0.5)
  - Color jitter (brightness, contrast, saturation, hue)

**Key Features**:
- Self-supervised learning: No labels required during training
- Strong data augmentation for creating positive pairs
- Temperature parameter controls the softness of the similarity distribution

## 🔬 Loss Functions

### Triplet Loss

```python
loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
```

Where:
- `d(x, y)` = L2 distance between embeddings (temperature-scaled)
- `margin`: Minimum desired separation between positive and negative pairs
- `temperature`: Scaling factor for embeddings before distance computation

**Hyperparameters**:
- **Margin**: Controls separation (typical values: 0.2 - 1.0)
- **Temperature**: Controls gradient scale (typical values: 0.5 - 1.0)

### NT-Xent Loss (SimCLR)

Normalized Temperature-scaled Cross Entropy Loss:
- Computes similarity matrix of all augmented pairs in batch
- Positive pairs: Two augmentations of the same image
- Negative pairs: Different images
- Uses cross-entropy to maximize similarity of positive pairs

**Hyperparameters**:
- **Temperature**: Controls softness of similarity distribution (typical values: 0.1 - 0.5)

## 🛠️ Key Features

### Data Handling

**TripletDataset**:
- Samples random triplets (anchor, positive, negative) on-the-fly
- Anchor: Random image from dataset
- Positive: Random image with same label as anchor
- Negative: Random image with different label from anchor

**SimCLRDataset**:
- Returns pairs of augmented views of the same image
- Uses `ContrastiveTransform` for data augmentation
- No labels required (self-supervised)

### Training Infrastructure

- **Progress tracking**: Real-time training progress with tqdm
- **Loss visualization**: Training and validation loss curves (linear and log scale)
- **Model checkpointing**: Saves models with hyperparameters (margin, temperature)
- **Validation**: Periodic validation during training
- **Embedding extraction**: Utilities to extract embeddings from trained models

### Evaluation & Visualization

**Embedding Analysis**:
- **PCA visualization**: 2D projections of embeddings
- **t-SNE visualization**: Non-linear dimensionality reduction
- **Clustering evaluation**: K-means clustering with Adjusted Rand Index (ARI)
- **Image-based visualization**: Embeddings displayed with actual face images

**Metrics**:
- Training/validation loss curves
- Clustering quality (ARI score)
- Embedding space visualization
- Distance analysis between embeddings

## 📁 Project Structure

```
Assignment6/
├── Assignment6.ipynb          # Main assignment notebook
├── Session6.ipynb             # Lab session materials
├── trainer.py                 # Training script with CLI
├── models.py                  # Model definitions (SiameseModel, SimCLR)
├── utils.py                   # Utility functions
│   ├── TripletDataset         # Dataset class for triplet sampling
│   ├── SimCLRDataset          # Dataset class for contrastive learning
│   ├── TripletLoss            # Triplet loss implementation
│   ├── Trainer                # Training loop for Siamese model
│   ├── NormLayer              # L2 normalization layer
│   ├── ContrastiveTransform   # Data augmentation for SimCLR
│   ├── nt_xent_loss           # NT-Xent loss for SimCLR
│   ├── get_embeddings         # Extract embeddings from models
│   ├── plot_both              # PCA/t-SNE visualization
│   ├── display_projections    # Embedding visualization
│   └── calculate_ARI          # Clustering evaluation
├── checkpoints/               # Saved model checkpoints
│   └── checkpoint_epoch_*_margin_*_temperature_*.pth
└── README.md
```

## 🚀 Usage

### Running the Notebook

1. **Install dependencies**:
   ```bash
   pip install torch torchvision numpy matplotlib seaborn tqdm scikit-learn scikit-image
   ```

2. **Open the notebook**:
   ```bash
   jupyter notebook Assignment6.ipynb
   ```

3. **Run cells sequentially** to:
   - Load and visualize the LFW dataset
   - Define models (SiameseModel, SimCLR)
   - Train models with different hyperparameters
   - Extract embeddings
   - Visualize embeddings (PCA, t-SNE)
   - Evaluate clustering performance
   - Compare model performance

### Training via Command Line

**Train Siamese Model**:
```bash
python trainer.py --model siamese --margin 1.0 --temperature 0.5 --n_iters 10000
```

**Train SimCLR Model**:
```bash
python trainer.py --model simclr --temperature 0.1
```

**Arguments**:
- `--model`: Model type (`siamese` or `simclr`)
- `--margin`: Margin for triplet loss (default: 1.0)
- `--temperature`: Temperature parameter (default: 0.5)
- `--n_iters`: Number of training iterations for Siamese (default: 10000)

### Loading Saved Models

```python
import torch
from models import SiameseModel, SimCLR
from utils import load_model

# Load Siamese model
model = SiameseModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
model, optimizer, epoch, stats = load_model(model, optimizer, 'checkpoints/checkpoint_epoch_10000_margin_1.0_temperature_0.5.pth')
```

### Extracting Embeddings

```python
from utils import get_embeddings

# For Siamese model
imgs_flat, embs, labels = get_embeddings(model, test_loader, device, simclr=False)

# For SimCLR model
imgs_flat, embs, labels = get_embeddings(model, test_loader, device, simclr=True)
```

### Visualizing Embeddings

```python
from utils import plot_both, display_projections

# PCA and t-SNE visualization
tsne_embs = plot_both(imgs_flat, embs, labels, target_names)

# Clustering evaluation
from utils import calculate_ARI
calculate_ARI(imgs_flat, embs, labels)
```

## 📈 Analysis & Results

<div align="center">
  <img src="imgs/Screenshot 2025-12-29 at 21.34.14.png" alt="Embedding visualization (t-SNE/PCA)" width="60%" height="60%">
  <br>
  <em>Projection and clustering of face embeddings learned with SimCLR/Triplet Loss</em>
</div>


### Model Comparison

The notebook includes comprehensive analysis:
- **Training curves**: Loss over iterations/epochs
- **Embedding quality**: PCA and t-SNE visualizations
- **Clustering performance**: ARI scores comparing raw images vs embeddings
- **Hyperparameter sensitivity**: Effect of margin and temperature
- **Distance analysis**: Embedding distances for same/different identities

<div align="center">
  <img src="imgs/Screenshot 2025-12-29 at 21.46.53.png" alt="SimCLR/Triplet Loss training curves" width="70%" height="80%">
  <br>
  <em>Embedding/feature representaion of the learnt latent-space</em>
</div>


### Key Findings

1. **Triplet Loss**: Effective for learning discriminative embeddings with proper margin selection
2. **SimCLR**: Self-supervised approach that learns useful representations without labels
3. **Temperature Scaling**: Critical hyperparameter affecting training stability and embedding quality
4. **Embedding Visualization**: t-SNE reveals clear clustering of same-identity faces
5. **Clustering Quality**: Embeddings achieve higher ARI scores than raw pixel clustering

### Hyperparameter Tuning

**TriNet Siamese**:
- **Margin**: Too small → insufficient separation, too large → training difficulty
- **Temperature**: Affects gradient scale and embedding distribution
- **Recommended**: margin=0.5-1.0, temperature=0.5-1.0

**SimCLR**:
- **Temperature**: Lower values → sharper similarity distribution
- **Recommended**: temperature=0.1-0.2 for face recognition

## 🔧 Utility Functions

### Core Classes

- **`TripletDataset`**: Samples triplets (anchor, positive, negative) from dataset
- **`SimCLRDataset`**: Returns pairs of augmented views for contrastive learning
- **`TripletLoss`**: Implements temperature-scaled triplet loss
- **`Trainer`**: Training loop for Siamese model with validation
- **`NormLayer`**: L2 normalization layer for embeddings
- **`ContrastiveTransform`**: Data augmentation pipeline for SimCLR

### Visualization Functions

- **`plot_both()`**: PCA and t-SNE visualization of images and embeddings
- **`display_projections()`**: Scatter plot of 2D projections with class colors
- **`display_projections_images()`**: Embedding visualization with actual face images
- **`visualize_progress()`**: Training/validation loss curves

### Evaluation Functions

- **`get_embeddings()`**: Extract embeddings from trained models
- **`calculate_ARI()`**: Compute Adjusted Rand Index for clustering evaluation
- **`smooth()`**: Smooth loss curves for visualization

## 🎓 Extra Credit

The assignment includes an extra credit component:

1. **Personal Photo Testing**:
   - Take photos of group members with different illuminations, angles, etc.
   - Extract embeddings using trained models
   - Compare embedding similarities between group members

2. **Celebrity Look-alike**:
   - Find which celebrity in LFW dataset has most similar embedding to your photos
   - Analyze embedding distances to celebrities
   - Visualize similarity rankings

## 🔗 References

- [TriNet Paper](https://arxiv.org/abs/1503.03832) - FaceNet: A Unified Embedding for Face Recognition and Clustering
- [SimCLR Paper](https://arxiv.org/abs/2002.05709) - A Simple Framework for Contrastive Learning of Visual Representations
- [LFW Dataset](http://vis-www.cs.umass.edu/lfw/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)

---

## 💬 Support

If you found this project helpful, you can support my work by buying me a coffee or via paypal!  

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-orange?logo=buy-me-a-coffee&style=for-the-badge)](https://buymeacoffee.com/amirhosseinsoltani)

[![PayPal](https://img.shields.io/badge/Donate%20with-PayPal-blue?logo=paypal&style=for-the-badge)](https://paypal.me/thesoltanic)



*This assignment demonstrates self-supervised and metric learning techniques for face recognition, comparing triplet-based and contrastive learning approaches.*

