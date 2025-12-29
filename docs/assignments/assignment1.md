# 1: Neural Network Fundamentals

## 📋 Overview

This assignment focuses on building and training basic neural networks from scratch on the CIFAR-10 dataset. The project implements and compares Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs), with experiments on regularization techniques and custom learning rate scheduling.

## 🎯 Objectives

- Implement and train MLP and CNN classifiers on CIFAR-10
- Compare model performance with and without dropout regularization
- Implement custom learning rate warmup scheduler
- Analyze learning curves, confusion matrices, and model predictions
- Investigate failure cases and model behavior

## 📊 Dataset

**CIFAR-10** - 10-class image classification dataset
- **Training samples**: 50,000
- **Test samples**: 10,000
- **Image size**: 32×32×3 (RGB)
- **Classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

The dataset is automatically downloaded using PyTorch's `torchvision.datasets.CIFAR10`.

## 🏗️ Models Implemented

### 1. Multi-Layer Perceptron (MLP)

A fully connected neural network with the following architecture:
- **Input**: Flattened 32×32×3 = 3,072 features
- **Hidden layers**: 
  - Layer 1: 3,072 → 1,024 (ReLU)
  - Layer 2: 1,024 → 512 (ReLU)
  - Layer 3: 512 → 256 (ReLU)
  - Layer 4: 256 → 128 (ReLU)
- **Output**: 128 → 10 (logits)
- **Total parameters**: ~3.8M

**Features**:
- Optional dropout layers (p=0.5) after each hidden layer
- Supports custom learning rate scheduling

### 2. Convolutional Neural Network (CNN)

A convolutional neural network with the following architecture:
- **Conv Block 1**: 3 → 32 → 64 channels (3×3 conv, ReLU, MaxPool)
- **Conv Block 2**: 64 → 128 → 128 channels (3×3 conv, ReLU, MaxPool)
- **Conv Block 3**: 128 → 256 → 256 channels (3×3 conv, ReLU, MaxPool)
- **Fully Connected**: 
  - 256×4×4 → 1,024 (ReLU)
  - 1,024 → 512 (ReLU)
  - 512 → 10 (logits)

**Features**:
- Optional dropout layers (p=0.5) after pooling and FC layers
- Padding to preserve spatial dimensions
- Progressive channel expansion

## 🔬 Experiments

The project includes 6 experiments comparing different configurations:

| Experiment | Model | Dropout | LR Scheduler | Description |
|------------|-------|---------|--------------|-------------|
| **Exp1** | MLP | ❌ | ❌ | Baseline MLP without regularization |
| **Exp2** | MLP | ✅ | ❌ | MLP with dropout (p=0.5) |
| **Exp3** | CNN | ❌ | ❌ | Baseline CNN without regularization |
| **Exp4** | CNN | ✅ | ❌ | CNN with dropout (p=0.5) |
| **Exp5** | MLP | ✅ | ✅ | MLP with dropout + custom LR scheduler |
| **Exp6** | CNN | ✅ | ✅ | CNN with dropout + custom LR scheduler |

### Training Configuration

All experiments use:
- **Optimizer**: Adam
- **Learning rate**: 0.0001
- **Batch size**: 1024
- **Epochs**: 100
- **Loss function**: CrossEntropyLoss
- **Validation**: Every 10 epochs

## 🛠️ Key Features

### Custom Learning Rate Warmup Scheduler

A custom linear warmup scheduler is implemented (no PyTorch schedulers used):

```python
def warmup_lr(optimizer, current_epoch, warmup_epochs, target_lr, init_lr=1e-6):
    """
    Linear warmup schedule: gradually increases LR from init_lr to target_lr
    over warmup_epochs, then maintains target_lr.
    """
```

**Parameters**:
- `warmup_epochs`: 25
- `init_lr`: 1e-6
- `target_lr`: 0.0001

### Training Infrastructure

- **TensorBoard logging**: Training/validation loss and learning rate curves
- **Model checkpointing**: Saves best models with training configurations
- **Progress tracking**: Real-time training progress with tqdm
- **Evaluation metrics**: Accuracy, confusion matrices, per-class performance

## 📁 Project Structure

```
Assignment1/
├── Assignment1.ipynb          # Main assignment notebook
├── Session1.ipynb             # Lab session materials
├── data/
│   ├── cifar-10-batches-py/   # CIFAR-10 dataset
│   └── MNIST/                 # MNIST dataset (if used)
├── models/
│   ├── Exp1/                  # Experiment 1 checkpoints
│   ├── Exp2/                  # Experiment 2 checkpoints
│   ├── Exp3/                  # Experiment 3 checkpoints
│   ├── Exp4/                  # Experiment 4 checkpoints
│   ├── Exp5/                  # Experiment 5 checkpoints
│   └── Exp6/                  # Experiment 6 checkpoints
├── log_dir/
│   ├── Exp1/                  # TensorBoard logs for Exp1
│   ├── Exp2/                  # TensorBoard logs for Exp2
│   └── ...                    # Logs for other experiments
└── imgs/                      # Visualization images
    ├── MLP.png
    ├── CNN.png
    ├── softmax.png
    └── ...
```

## 📈 Analysis & Results

### Model Comparison

The notebook includes comprehensive analysis:
- **Learning curves**: Training vs validation loss over epochs
- **Confusion matrices**: Per-class classification performance
- **Accuracy metrics**: Overall and per-class accuracy
- **Failure case analysis**: Visualization of misclassified images
- **Overfitting analysis**: Comparison of models with/without dropout

### Key Findings

1. **Dropout Regularization**: Reduces overfitting gap between training and validation loss
2. **Learning Rate Scheduling**: Custom warmup helps stabilize training in early epochs
3. **CNN vs MLP**: CNNs generally outperform MLPs on image classification tasks
4. **Failure Cases**: Models struggle with similar classes (e.g., cat vs dog, bird vs airplane)

## 🚀 Usage

### Running the Notebook

1. **Install dependencies**:
   ```bash
   pip install torch torchvision numpy matplotlib seaborn tqdm pyyaml tensorboard torchmetrics
   ```

2. **Open the notebook**:
   ```bash
   jupyter notebook Assignment1.ipynb
   ```

3. **Run experiments**: Execute cells sequentially to:
   - Download and inspect the dataset
   - Define models (MLP and CNN)
   - Train experiments (Exp1-Exp6)
   - Evaluate models and visualize results

### Viewing TensorBoard Logs

```bash
tensorboard --logdir=log_dir
```

Then open `http://localhost:6006` in your browser to view training curves.

### Loading Saved Models

```python
checkpoint = torch.load('models/Exp1/checkpoint_Exp1.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

## 🔗 References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorBoard](https://www.tensorflow.org/tensorboard)

---

---

## 💬 Support

If you found this project helpful, you can support my work by buying me a coffee or via paypal!  

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-orange?logo=buy-me-a-coffee&style=for-the-badge)](https://buymeacoffee.com/amirhosseinsoltani)

[![PayPal](https://img.shields.io/badge/Donate%20with-PayPal-blue?logo=paypal&style=for-the-badge)](https://paypal.me/thesoltanic)


## Location

The complete assignment documentation, code, and notebooks are located in:
```
src/Assignment1/
```
---
*This assignment demonstrates fundamental deep learning concepts including neural network architectures, regularization techniques, and training optimization strategies.*