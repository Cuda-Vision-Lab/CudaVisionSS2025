# 3: Recurrent Neural Networks for Action Recognition

## 📋 Overview

This assignment focuses on implementing Recurrent Neural Networks (RNNs) from scratch and applying them to action recognition tasks. The project implements custom LSTM and Convolutional LSTM cells, then compares them with PyTorch's built-in RNN modules on the KTH-Actions dataset for video action classification.

<div align="center">
  <img src="https://github.com/Cuda-Vision-Lab/CudaVisionSS2025/raw/main/src/Assignment3/imgs/Screenshot%202025-12-29%20at%2021.10.39.png" alt="Sample LSTM output for action recognition" width="70%" height="80%">
  <br>
  <em>Sample output from the RNN action recognition model</em>
</div>


## 🎯 Objectives

- Implement LSTM and ConvLSTM cells from scratch
- Build an action recognition pipeline using RNNs
- Compare different RNN architectures (LSTMCell, GRUCell, custom implementations)
- Evaluate models on accuracy, training/inference time, and parameter count
- Implement 3D-CNN (R(2+1)d-Net) for action classification (extra credit)

## 📊 Dataset

**KTH-Actions Dataset** - Human action recognition dataset
- **Actions**: walking, jogging, running, boxing, handwaving, handclapping
- **Frame size**: 64×64 pixels (grayscale)
- **Sequence length**: 10 frames per sample
- **Split**: Person IDs 0-16 for training, 17-25 for testing
- **Source**: [KTH-Actions Dataset](https://www.csc.kth.se/cvap/actions/)

The dataset is automatically loaded using the custom `KTHActionDataset` class in `src/dataloader.py`.

## 🏗️ Models Implemented

### 1. Custom LSTM (OwnLSTM)

A fully custom LSTM implementation from scratch with the following components:

**Architecture**:
- **Forget Gate**: `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`
- **Input Gate**: `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)`
- **Candidate Gate**: `C̃_t = tanh(W_C · [h_{t-1}, x_t] + b_C)`
- **Output Gate**: `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)`
- **Cell State**: `C_t = f_t ⊙ C_{t-1} + i_t ⊙ C̃_t`
- **Hidden State**: `h_t = o_t ⊙ tanh(C_t)`

**Features**:
- Xavier weight initialization
- Supports both single-step and sequence inputs
- Custom forward pass implementation
- Final linear layer for classification output

### 2. Convolutional LSTM Cell (ConvLSTMCell)

A convolutional variant of LSTM that preserves spatial information:

**Architecture**:
- Uses 1D convolutions instead of linear layers
- Maintains spatial dimensions through the sequence
- Separate convolutional layers for each gate (forget, input, candidate, output)
- Kernel size: 3 (default), with padding to preserve dimensions

**Features**:
- Processes spatial-temporal data efficiently
- Suitable for video sequences with spatial structure
- Custom implementation matching standard ConvLSTM formulation

### 3. Action Classifier

A complete action recognition model with three main components:

#### Encoder
- **Option 1**: Custom CNN encoder
  - 5 convolutional blocks with BatchNorm and GELU activation
  - Progressive channel expansion: 1 → 16 → 32 → 64 → 128 → emb_dim
  - Adaptive average pooling to fixed size
  
- **Option 2**: Pretrained ResNet18 encoder
  - Modified first layer for grayscale input (1 channel)
  - Feature extraction with projection to embedding dimension

#### Recurrent Module
Supports multiple RNN architectures:
- **LSTMCell**: PyTorch's built-in LSTM cell
- **GRUCell**: PyTorch's built-in GRU cell
- **OwnLSTM**: Custom LSTM implementation
- **OwnConvLSTM**: Custom ConvLSTM implementation

#### Classifier
- Conv1d layer for temporal feature extraction
- Adaptive average pooling
- Fully connected layer for final classification (6 classes)

## 🔬 Experiments

The project includes multiple experiments comparing different RNN architectures:

| Experiment | RNN Type | Pretrained Encoder | Scheduler | Description |
|------------|----------|-------------------|-----------|-------------|
| **LSTMCell** | PyTorch LSTM | ❌ | ✅ | Baseline with PyTorch LSTM |
| **LSTMCell_NoScheduler** | PyTorch LSTM | ❌ | ❌ | LSTM without learning rate scheduling |
| **GRUCell** | PyTorch GRU | ❌ | ✅ | GRU-based model |
| **GRUCell_NoScheduler** | PyTorch GRU | ❌ | ❌ | GRU without scheduling |
| **OwnLSTM** | Custom LSTM | ❌ | ✅ | Custom LSTM implementation |
| **LSTMCell_PretEncoder** | PyTorch LSTM | ✅ | ❌ | LSTM with pretrained ResNet encoder |
| **LSTMCell_PretEncoder_Scheduler** | PyTorch LSTM | ✅ | ✅ | LSTM with pretrained encoder + scheduler |

### Training Configuration

All experiments use:
- **Optimizer**: Adam
- **Learning rate**: 0.001 (with optional scheduler)
- **Batch size**: 32
- **Epochs**: 50-100 (varies by experiment)
- **Loss function**: CrossEntropyLoss
- **Embedding dimension**: 128
- **Hidden dimension**: 128
- **Number of layers**: 2

## 🛠️ Key Features

### Data Augmentation

**Spatial Augmentations**:
- Random horizontal flip (p=0.5)
- Random rotation (±25 degrees)

**Temporal Augmentations**:
- Random temporal sampling (slicing step)
- Random temporal reversal (p=0.3)

### Training Infrastructure

- **TensorBoard logging**: Training/validation loss, accuracy, and learning rate curves
- **Model checkpointing**: Saves best models with training configurations
- **Progress tracking**: Real-time training progress with tqdm
- **Evaluation metrics**: Accuracy, per-class performance
- **Experiment management**: YAML configuration files for each experiment

### Custom Utilities

- **Seed management**: Reproducible experiments
- **Model evaluation**: Comprehensive evaluation functions
- **Visualization**: Sequence visualization tools
- **Data loading**: Efficient dataset handling with proper train/test splits

## 📁 Project Structure

```
Assignment3/
├── Assignment3.ipynb          # Main assignment notebook
├── session3.ipynb             # Lab session materials
├── src/
│   ├── models.py              # Custom LSTM and ConvLSTM implementations
│   ├── dataloader.py          # KTHActionDataset class
│   ├── transformations.py     # Data augmentation transforms
│   ├── utils.py               # Training and evaluation utilities
│   └── devel/
│       ├── task1.ipynb        # Task 1 development notebook
│       ├── task2.ipynb        # Task 2 development notebook
│       └── task3.ipynb        # Task 3 (extra credit) notebook
├── data/
│   └── README.md              # Dataset information
├── models/
│   └── README.md              # Model checkpoints directory
├── tboard_logs/               # TensorBoard logs for all experiments
│   ├── LSTMCell/
│   ├── GRUCell/
│   ├── OwnLSTM/
│   └── ...
└── imgs/                      # Visualization images and GIFs
    ├── pipeline.png
    ├── gif_*.gif
    └── ...
```

## 📈 Analysis & Results

### Model Comparison

The notebook includes comprehensive analysis:
- **Learning curves**: Training vs validation loss and accuracy over epochs
- **Performance metrics**: Overall and per-class accuracy
- **Parameter count**: Comparison of model sizes
- **Training/inference time**: Efficiency analysis
- **Failure case analysis**: Visualization of misclassified sequences

### Key Findings

1. **GRU Performance**: GRUCell achieved the best performance on the dataset
2. **LSTM vs GRU**: GRU's simpler architecture (no cell state) can be more efficient while maintaining performance
3. **Custom Implementation**: OwnLSTM showed competitive results, validating the implementation
4. **Pretrained Encoders**: Using pretrained ResNet encoders improved feature extraction
5. **Learning Rate Scheduling**: Schedulers helped stabilize training and improve convergence
6. **Temporal Augmentations**: Effective for improving generalization

## 🚀 Usage

### Running the Notebook

1. **Install dependencies**:
   ```bash
   pip install torch torchvision numpy matplotlib seaborn tqdm pyyaml tensorboard pillow
   ```

2. **Download the KTH-Actions dataset**:
   - The dataset should be placed in the appropriate directory
   - Or modify the `root_dir` parameter in `KTHActionDataset`

3. **Open the notebook**:
   ```bash
   jupyter notebook Assignment3.ipynb
   ```

4. **Run experiments**: Execute cells sequentially to:
   - Implement custom LSTM and ConvLSTM cells (Task 1)
   - Load and preprocess the KTH-Actions dataset
   - Train different RNN architectures (Task 2)
   - Evaluate and compare models
   - Visualize results

### Viewing TensorBoard Logs

```bash
tensorboard --logdir=tboard_logs
```

Then open `http://localhost:6006` in your browser to view training curves for all experiments.

### Loading Saved Models

```python
checkpoint = torch.load('models/experiment_name/checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Using Custom Models

```python
from src.models import OwnLSTM, ConvLSTMCell
from src.dataloader import KTHActionDataset
from src.transformations import get_train_transforms, get_test_transforms

# Initialize custom LSTM
lstm = OwnLSTM(input_size=128, hidden_size=128, output_size=128)

# Load dataset
train_dataset = KTHActionDataset(
    root_dir='path/to/kth_actions',
    split='train',
    transform=get_train_transforms(slicing_step=2),
    max_frames=10,
    img_size=(64, 64)
)
```

## 🎓 Extra Credit: 3D-CNN Implementation

The project includes an implementation of R(2+1)d-Net for action recognition:

- **Architecture**: Factorized 3D convolutions (2D spatial + 1D temporal)
- **Advantages**: More efficient than full 3D convolutions while maintaining performance
- **Comparison**: Evaluated against RNN-based models

See `src/devel/task3.ipynb` for implementation details.

## 🔗 References

- [KTH-Actions Dataset](https://www.csc.kth.se/cvap/actions/)
- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Convolutional LSTM Network](https://arxiv.org/abs/1506.04214)
- [R(2+1)D Networks](https://arxiv.org/abs/1711.11248)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [TensorBoard](https://www.tensorflow.org/tensorboard)


*Date: 18.05.2025*

---

## 💬 Support

If you found this project helpful, you can support my work by buying me a coffee or via PayPal!

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-orange?logo=buy-me-a-coffee&style=for-the-badge)](https://buymeacoffee.com/amirhosseinsoltani)

[![PayPal](https://img.shields.io/badge/Donate%20with-PayPal-blue?logo=paypal&style=for-the-badge)](https://paypal.me/thesoltanic)

---

## Location

The complete assignment documentation, code, and notebooks are located in:
```
src/Assignment3/
```

---
*This assignment demonstrates deep understanding of recurrent neural networks, including custom implementations of LSTM and ConvLSTM cells, and their application to video action recognition tasks.*