# 4: Variational Autoencoders (VAEs)

## 📋 Overview

This assignment focuses on implementing and training Variational Autoencoders (VAEs) for image generation and reconstruction. The project implements both standard Convolutional VAEs (CVAE) and Conditional Convolutional VAEs (CCVAE) on the AFHQ (Animal Faces-High Quality) dataset, with experiments exploring the effect of KL divergence weighting on model performance.
<p align="center">
  <img src="https://github.com/Cuda-Vision-Lab/CudaVisionSS2025/raw/main/src/Assignment4/output.gif" alt="VAE Results Animation"/>
</p>

## 🎯 Objectives

- Implement Convolutional Variational Autoencoder (CVAE) from scratch
- Implement Conditional Convolutional Variational Autoencoder (CCVAE) with class conditioning
- Understand the reparameterization trick and ELBO (Evidence Lower BOund) optimization
- Experiment with different KL divergence weights (λ_KLD) to balance reconstruction and regularization
- Visualize latent space representations and generate novel images
- Analyze the trade-off between reconstruction quality and latent space regularization

## 📊 Dataset

**AFHQ (Animal Faces-High Quality)** - High-quality animal face dataset
- **Training samples**: ~15,000 images
- **Test samples**: ~1,500 images
- **Image size**: 64×64×3 (RGB)
- **Classes**: 3 categories (cat, dog, wildlife)
- **Download**: Automatically downloaded via `download.sh` script

The dataset is organized in ImageFolder format with train/test splits.

## 🏗️ Models Implemented

### 1. Convolutional VAE (CVAE)

A standard Variational Autoencoder with convolutional encoder-decoder architecture:

**Encoder**:
- Input: (3, 64, 64)
- Conv layers: 3→16→32→64→128 channels
- Output: 2048-dimensional flattened features
- Fully connected layers: 2048 → μ, log(σ²) (latent_dim dimensions)

**Latent Space**:
- Reparameterization trick: z = μ + ε·σ, where ε ~ N(0,1)
- Default latent dimension: 64 (configurable)

**Decoder**:
- Input projection: latent_dim → 2048
- Reshape: 2048 → (128, 4, 4)
- Deconv layers: 128→64→32→16→3 channels
- Output: (3, 64, 64) with Sigmoid activation

**Key Features**:
- Batch normalization after each conv/deconv layer
- LeakyReLU(0.2) activations
- Reparameterization trick for differentiable sampling
- MSE reconstruction loss + KL divergence regularization

### 2. Conditional Convolutional VAE (CCVAE)

An extension of CVAE that conditions both encoder and decoder on class labels:

**Architecture**:
- Same encoder-decoder structure as CVAE
- **Class conditioning**: One-hot encoded class labels concatenated with:
  - Encoder output (before μ, log(σ²) computation)
  - Latent vector (before decoder input projection)
- Supports controlled generation by specifying class labels

**Key Features**:
- Class-conditional encoding and decoding
- Same architecture as CVAE with additional class embeddings
- Enables class-specific image generation
- Useful for controlled generation tasks

## 🔬 Experiments

The project includes multiple experiments exploring different KL divergence weights:

| Experiment | Model | λ_KLD | Latent Dim | Description |
|------------|-------|-------|------------|-------------|
| **CVAE1** | CVAE | 0.001 | 64 | Baseline with moderate KL weight |
| **CVAE2** | CVAE | 0.0 | 64 | No KL regularization (pure autoencoder) |
| **CVAE3** | CVAE | 0.01 | 64 | Higher KL weight (stronger regularization) |
| **CVAE4** | CVAE | 0.0001 | 64 | Lower KL weight (weaker regularization) |
| **CVAE_new1** | CVAE | 0.0001 | 64 | Variant with different architecture |
| **CCVAE1** | CCVAE | 0.0001 | 64 | Conditional VAE with class labels |

### Training Configuration

Default training parameters:
- **Optimizer**: AdamW
- **Learning rate**: 0.001 (configurable)
- **Batch size**: 64
- **Epochs**: 50
- **Weight decay**: 1e-4
- **Scheduler**: ReduceLROnPlateau (patience=7, factor=0.5)
- **Loss function**: MSE reconstruction + λ_KLD × KL divergence

### Loss Function

The VAE loss combines reconstruction and regularization terms:

```python
Loss = MSE(reconstruction, target) + λ_KLD × KL(q(z|x) || p(z))
```

Where:
- **Reconstruction loss**: Mean Squared Error between input and reconstructed images
- **KL divergence**: Regularization term encouraging latent distribution to match prior N(0,I)
- **λ_KLD**: Weight controlling the trade-off between reconstruction quality and latent space regularization

## 🛠️ Key Features

### Training Infrastructure

- **TensorBoard logging**: Training/validation loss, reconstruction loss, KL divergence, learning rate
- **Image visualization**: Automatic saving of reconstruction comparisons every N epochs
- **Model checkpointing**: Saves model states with training statistics
- **Progress tracking**: Real-time training progress with tqdm progress bars
- **Config management**: YAML-based configuration files for experiment reproducibility

### Visualization Tools

- **Reconstruction comparison**: Side-by-side original vs reconstructed images
- **Latent space visualization**: PCA projection of latent representations colored by class
- **Image generation**: Sample from latent space to generate novel images
- **Latent space traversal**: Visualize how changes in latent dimensions affect generated images

### Utility Functions

- `denormalize_images()`: Convert images from [-1, 1] to [0, 1] range
- `vae_loss_function()`: Combined reconstruction and KL divergence loss
- `train_model()`: Complete training loop with validation
- `eval_model()`: Model evaluation with image saving
- `vis_latent()`: Visualize latent space using PCA
- `inference()`: Generate images from random latent vectors
- `save_model()` / `load_model()`: Model checkpoint management

## 📁 Project Structure

```
Assignment4/
├── Assignment4.ipynb          # Main assignment notebook
├── Session4.ipynb             # Lab session materials
├── cvae.py                    # CVAE model implementation
├── ccvae.py                   # CCVAE model implementation
├── trainer.py                 # Training script
├── utils.py                   # Utility functions (training, evaluation, visualization)
├── download.sh                # Dataset download script
├── configs/                   # Experiment configurations
│   ├── CVAE1_KLD_0.001/
│   ├── CVAE2_KLD_0.0/
│   ├── CVAE3_KLD_0.01/
│   ├── CVAE4_KLD_0.0001/
│   ├── CVAE_new1_KLD_0.0001/
│   └── CCVAE1_KLD_0.0001/
├── data/
│   └── AFHQ/                  # AFHQ dataset (downloaded)
│       ├── train/
│       └── test/
├── models/                    # Saved model checkpoints
├── imgs/                      # Generated images and visualizations
│   ├── CVAE1/                 # Experiment outputs
│   ├── CVAE2/
│   ├── inference/             # Generated samples
│   └── ...
├── tboard_logs/               # TensorBoard log files
│   ├── CVAE1_KLD_0.001/
│   └── ...
└── htmls/                     # HTML exports of notebooks
```

## 📈 Analysis & Results

### KL Divergence Weight Analysis

The λ_KLD parameter controls the trade-off between:
- **Reconstruction quality**: Lower λ_KLD → better reconstruction, but less regularized latent space
- **Latent space structure**: Higher λ_KLD → more structured latent space, but potentially worse reconstruction

**Key Findings**:
1. **λ_KLD = 0.0**: Acts as a pure autoencoder, excellent reconstruction but unstructured latent space
2. **λ_KLD = 0.0001**: Weak regularization, good reconstruction with some latent structure
3. **λ_KLD = 0.001**: Balanced trade-off (default)
4. **λ_KLD = 0.01**: Strong regularization, well-structured latent space but may sacrifice reconstruction quality

### Latent Space Properties

- **Disentanglement**: Higher KL weights encourage more disentangled representations
- **Interpolation**: Well-regularized latent spaces enable smooth interpolation between samples
- **Generation quality**: Conditional VAEs enable class-specific generation with better control

## 🚀 Usage

### Setup

1. **Install dependencies**:
   ```bash
   pip install torch torchvision numpy matplotlib tqdm pyyaml tensorboard scikit-learn
   ```

2. **Download dataset**:
   ```bash
   chmod +x download.sh
   ./download.sh
   ```
   This will download and extract the AFHQ dataset to `./data/AFHQ/`

### Training a Model

#### Using the Training Script

```python
from trainer import main
from cvae import CVAE

configs = {
    "model_name": "CVAE",
    "exp": "1",
    "latent_dim": 64,
    "batch_size": 64,
    "num_epochs": 50,
    "lr": 0.001,
    "scheduler": "ReduceLROnPlateau",
    "use_scheduler": True,
    "lambda_kld": 0.001,
}

main(configs)
```

#### Using the Notebook

1. Open `Assignment4.ipynb` in Jupyter
2. Run cells sequentially to:
   - Load and inspect the dataset
   - Define and initialize models
   - Train experiments with different configurations
   - Evaluate models and visualize results
   - Generate and analyze samples

### Viewing TensorBoard Logs

```bash
tensorboard --logdir=tboard_logs
```

Then open `http://localhost:6006` in your browser to view:
- Training/validation loss curves
- Reconstruction vs KL divergence components
- Learning rate schedule
- Image reconstructions

### Loading and Using Trained Models

```python
import torch
from cvae import CVAE
from utils import load_model

# Initialize model
model = CVAE(latent_dim=64)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

# Load checkpoint
model, optimizer, epoch, stats = load_model(
    model, optimizer, 
    'models/CVAE1/checkpoint_KLD_0.001_epoch_49.pth'
)

# Generate samples
model.eval()
with torch.no_grad():
    z = torch.randn(16, 64).to(device)
    z = model.decoder_input(z)
    z = z.view(-1, 128, 4, 4)
    samples = model.decoder(z)
```

### Conditional Generation (CCVAE)

```python
from ccvae import CCVAE

model = CCVAE(latent_dim=64, num_classes=3)

# Generate samples for specific class (0=cat, 1=dog, 2=wildlife)
class_label = torch.tensor([0] * 16)  # Generate 16 cat faces
samples = model.sample(num_samples=16, c=class_label)
```

## 🔧 Configuration Files

Each experiment has a YAML configuration file in `configs/`:

```yaml
batch_size: 64
exp: '1'
lambda_kld: 0.001
latent_dim: 64
lr: 0.001
model_name: CVAE
num_epochs: 50
scheduler: ReduceLROnPlateau
use_scheduler: true
```

## 📝 Key Concepts

### Variational Autoencoder

A VAE is a generative model that learns to encode data into a latent distribution and decode samples from that distribution. Unlike standard autoencoders, VAEs learn a probabilistic latent representation.

### Reparameterization Trick

Enables backpropagation through random sampling:
```
z = μ + ε · σ, where ε ~ N(0,1)
```
This makes the sampling process differentiable.

### ELBO (Evidence Lower BOund)

The VAE objective function:
```
ELBO = E[log p(x|z)] - KL(q(z|x) || p(z))
```
Maximizing ELBO is equivalent to maximizing the data likelihood while regularizing the latent distribution.

### KL Divergence

Measures how different the learned latent distribution q(z|x) is from the prior p(z) = N(0,I). Encourages the encoder to produce latent codes that match the standard normal distribution.

## 🔗 References

- [Auto-Encoding Variational Bayes (Kingma & Welling, 2014)](https://arxiv.org/abs/1312.6114)
- [AFHQ Dataset](https://github.com/clovaai/stargan-v2)
- [PyTorch VAE Tutorial](https://github.com/pytorch/examples/tree/main/vae)
- [TensorBoard Documentation](https://www.tensorflow.org/tensorboard)

---

## 💬 Support

If you found this project helpful, you can support my work by buying me a coffee or via paypal!  

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-orange?logo=buy-me-a-coffee&style=for-the-badge)](https://buymeacoffee.com/amirhosseinsoltani)

[![PayPal](https://img.shields.io/badge/Donate%20with-PayPal-blue?logo=paypal&style=for-the-badge)](https://paypal.me/thesoltanic)

---

## Location

The complete assignment documentation, code, and notebooks are located in:
```
src/Assignment4/
```
---
*This assignment demonstrates variational inference, generative modeling, and the trade-offs between reconstruction quality and latent space regularization in deep learning.*
