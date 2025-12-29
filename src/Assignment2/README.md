# Assignment 2: Transfer Learning and Fine-tuning

## 📋 Overview

This assignment focuses on leveraging pre-trained deep learning models for a custom binary classification task: distinguishing between humans and robots. The project explores different transfer learning strategies, comparing multiple state-of-the-art architectures and fine-tuning approaches.

<div align="center">
  <img src="imgs/Screenshot 2025-12-29 at 21.05.50.png" alt="Sample human/robot classifier output" width="70%" height="80%">
  <br>
  <em>Sample output from the human/robot classifier</em>
</div>


## 🎯 Objectives

- Fine-tune pre-trained models (ResNet18, ConvNeXt, EfficientNet) for human/robot classification
- Compare different transfer learning strategies:
  - Full fine-tuning
  - Fixed feature extractor
  - Combined approach (partial fine-tuning)
- Evaluate model performance and analyze results
- Explore transformer-based architectures (DINOv2, SwinTransformer) for the classification task

## 📊 Dataset

**Human/Robot Binary Classification Dataset**
- **Task**: Binary classification (Human vs Robot)
- **Training samples**: 300 images
- **Validation samples**: 60 images
- **Test samples**: 42 images
- **Unseen Robot set**: 42 images (for additional evaluation)
- **Image size**: 224×224×3 (RGB)

The dataset is organized in ImageFolder structure:
```
dataset/
├── train/
│   ├── human/
│   └── robot/
├── val/
│   ├── human/
│   └── robot/
└── test/
    ├── human/
    └── robot/
```

## 🏗️ Models Implemented

### 1. Convolutional Neural Networks (CNNs)

#### ResNet18
- Pre-trained on ImageNet
- Architecture: Residual blocks with skip connections
- Fine-tuned for binary classification

#### ConvNeXt (Tiny)
- Modern CNN architecture inspired by Vision Transformers
- Pre-trained on ImageNet
- Architecture: 768-dimensional feature space
- Custom classifier head: 768 → 512 → 2

#### EfficientNet-B0
- Efficient architecture with compound scaling
- Pre-trained on ImageNet
- Optimized for accuracy and efficiency trade-off

### 2. Vision Transformers

#### DINOv2 (Small)
- Self-supervised vision transformer
- Pre-trained on large-scale unlabeled data
- Architecture: ViT-Small with patch size 14
- Feature dimension: 384

#### Swin Transformer
- Hierarchical vision transformer
- Shifted window-based self-attention
- Pre-trained on ImageNet

## 🔬 Transfer Learning Strategies

The project compares three different approaches to transfer learning:

### 1. Full Fine-tuning
- **Description**: All pre-trained model parameters are trainable
- **Approach**: Unfreeze all layers and train with lower learning rate
- **Use case**: When you have sufficient data and computational resources
- **Advantages**: Can adapt all features to the target task
- **Disadvantages**: Risk of overfitting, requires more data

### 2. Fixed Feature Extractor
- **Description**: Pre-trained backbone is frozen, only classifier head is trained
- **Approach**: Set `requires_grad=False` for all backbone parameters
- **Use case**: Limited data or computational resources
- **Advantages**: Fast training, prevents overfitting, preserves pre-trained features
- **Disadvantages**: Limited adaptation to target domain

### 3. Combined Approach (Partial Fine-tuning)
- **Description**: Freeze early layers, fine-tune later layers + classifier
- **Approach**: Freeze early feature extraction layers, unfreeze deeper layers
- **Use case**: Balance between adaptation and overfitting prevention
- **Advantages**: Better adaptation than fixed extractor, less overfitting than full fine-tuning
- **Disadvantages**: Requires careful selection of which layers to freeze

## 🛠️ Key Features

### Data Preprocessing

**Normalization**:
- Calculated dataset-specific mean and std from training + validation data
- Mean: `[0.4704, 0.4458, 0.4169]`
- Std: `[0.2250, 0.2159, 0.2180]`
- Prevents data leakage by excluding test set from normalization calculation

**Data Augmentation** (Training):
- Random horizontal flip
- Random rotation (±15 degrees)
- Color jitter (brightness=0.2, contrast=0.2)
- Resize to 224×224
- Normalization

**Validation/Test**:
- Resize to 224×224
- Normalization (no augmentation)

### Training Infrastructure

- **TensorBoard logging**: Training/validation loss, accuracy, and learning rate curves
- **Model checkpointing**: Saves best models with training configurations
- **Progress tracking**: Real-time training progress with tqdm
- **Evaluation metrics**: Accuracy, confusion matrices, per-class performance
- **Learning rate scheduling**: StepLR scheduler (decay by factor of 1/3 every 5 epochs)

### Training Configuration

- **Optimizer**: Adam
- **Learning rate**: 1e-4
- **Batch size**: 16
- **Loss function**: CrossEntropyLoss
- **Scheduler**: StepLR (step_size=5, gamma=1/3)

## 📁 Project Structure

```
Assignment2/
├── src/
│   ├── Assignment2.ipynb          # Main assignment notebook
│   ├── session2.ipynb              # Lab session materials
│   ├── dataset_downloader.ipynb    # Dataset download script
│   ├── utils.py                    # Utility functions (training, evaluation, visualization)
│   ├── devel/
│   │   ├── task1.ipynb             # Task 1: CNN fine-tuning experiments
│   │   ├── task2.ipynb             # Task 2: Transformer experiments
│   │   └── task3.ipynb             # Task 3: Additional experiments
│   └── tboard_logs/
│       ├── Task1_Logs/
│       │   ├── ResNet18_Tuned/
│       │   ├── ConvNext_Tuned/
│       │   ├── ConvNext_Fixed_Feature_Extractor/
│       │   ├── ConvNext_Combined_Approach/
│       │   └── EfficientNet_Tuned/
│       ├── Transformer/
│       │   ├── DINOv2/
│       │   └── SwinTransformer/
│       └── test/                   # Experimental logs
├── imgs/                           # Visualization images
│   ├── loss_1.png
│   ├── loss_2.png
│   ├── matrix.png
│   ├── matrix_nice.png
│   ├── train_eval.png
│   └── reference.png
└── README.md
```

## 📈 Analysis & Results

### Model Comparison

The notebook includes comprehensive analysis:
- **Learning curves**: Training vs validation loss over epochs
- **Confusion matrices**: Per-class classification performance
- **Accuracy metrics**: Overall and per-class accuracy
- **Model comparison**: Performance comparison across different architectures
- **Transfer learning comparison**: Comparison of fine-tuning strategies

### Key Findings

1. **Transfer Learning Effectiveness**: Pre-trained models significantly outperform training from scratch
2. **Architecture Comparison**: Different architectures show varying performance on the human/robot task
3. **Fine-tuning Strategy**: Combined approach often provides best balance between performance and overfitting
4. **Feature Extraction**: Fixed feature extractor is effective for small datasets
5. **Transformer Models**: Vision transformers (DINOv2, Swin) show competitive performance

## 🚀 Usage

### Setup

1. **Install dependencies**:
   ```bash
   pip install torch torchvision numpy matplotlib seaborn tqdm pyyaml tensorboard torchmetrics timm
   ```

2. **Download dataset**:
   - Run `dataset_downloader.ipynb` to download and organize the dataset
   - Or manually organize images into `dataset/train/`, `dataset/val/`, and `dataset/test/` folders

3. **Open the notebook**:
   ```bash
   jupyter notebook src/Assignment2.ipynb
   ```

### Running Experiments

1. **Data Preparation**:
   - Calculate dataset statistics (mean/std)
   - Set up data loaders with appropriate transforms

2. **Model Training**:
   - Load pre-trained models
   - Modify classifier heads for binary classification
   - Choose transfer learning strategy (fine-tuning/fixed/combined)
   - Train models with TensorBoard logging

3. **Evaluation**:
   - Evaluate on test set
   - Generate confusion matrices
   - Visualize results

### Viewing TensorBoard Logs

```bash
tensorboard --logdir=src/tboard_logs
```

Then open `http://localhost:6006` in your browser to view training curves.

### Loading Saved Models

```python
checkpoint = torch.load('models/checkpoint_ResNet18_Tuned.pth')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

## 🔧 Utility Functions

The `utils.py` file provides:

- `train_epoch()`: Training for one epoch
- `eval_model()`: Model evaluation on validation/test set
- `train_model()`: Complete training loop with TensorBoard logging
- `save_model()` / `load_model()`: Model checkpointing
- `plot()`: Visualization of training curves
- `plot_cm_matrix()`: Confusion matrix visualization
- `smooth()`: Loss curve smoothing
- `set_random_seed()`: Reproducibility utilities

## 🔗 References

- [Transfer Learning Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [ConvNeXt Paper](https://arxiv.org/abs/2201.03545)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)
- [DINOv2 Paper](https://arxiv.org/abs/2304.07193)
- [Swin Transformer Paper](https://arxiv.org/abs/2103.14030)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

---

## 💬 Support

If you found this project helpful, you can support my work by buying me a coffee or via paypal!  

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-orange?logo=buy-me-a-coffee&style=for-the-badge)](https://buymeacoffee.com/amirhosseinsoltani)

[![PayPal](https://img.shields.io/badge/Donate%20with-PayPal-blue?logo=paypal&style=for-the-badge)](https://paypal.me/thesoltanic)



*This assignment demonstrates transfer learning techniques, comparing different fine-tuning strategies and state-of-the-art architectures for computer vision tasks.*

