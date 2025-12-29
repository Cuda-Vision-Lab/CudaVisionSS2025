# Repository Structure

This page provides links to the comprehensive documentation for each assignment and the course project. Each assignment directory contains a detailed README file with complete information about implementations, experiments, and usage.

## Assignments

### Assignment 1: Neural Network Fundamentals

**Location**: `src/Assignment1/`

**Description**: Building and training basic neural networks from scratch on the CIFAR-10 dataset. Implements and compares Multi-Layer Perceptrons (MLPs) and Convolutional Neural Networks (CNNs), with experiments on regularization techniques and custom learning rate scheduling.

**Documentation**: See [Assignment 1 README](https://github.com/Cuda-Vision-Lab/CudaVisionSS2025/blob/main/src/Assignment1/README.md)

**Key Topics**:
- MLP and CNN architectures
- Dropout regularization
- Custom learning rate warmup scheduler
- Model evaluation and analysis

---

### Assignment 2: Transfer Learning and Fine-tuning

**Location**: `src/Assignment2/`

**Description**: Leveraging pre-trained deep learning models for a custom binary classification task (human vs robot). Explores different transfer learning strategies and compares multiple state-of-the-art architectures.

**Documentation**: See [Assignment 2 README](https://github.com/Cuda-Vision-Lab/CudaVisionSS2025/blob/main/src/Assignment2/README.md)

**Key Topics**:
- Transfer learning strategies (full fine-tuning, fixed feature extractor, combined approach)
- Pre-trained models: ResNet18, ConvNeXt, EfficientNet
- Vision transformers: DINOv2, SwinTransformer
- Model comparison and evaluation

---

### Assignment 3: Recurrent Neural Networks for Action Recognition

**Location**: `src/Assignment3/`

**Description**: Implementing Recurrent Neural Networks (RNNs) from scratch and applying them to action recognition tasks. Implements custom LSTM and Convolutional LSTM cells, then compares them with PyTorch's built-in RNN modules.

**Documentation**: See [Assignment 3 README](https://github.com/Cuda-Vision-Lab/CudaVisionSS2025/blob/main/src/Assignment3/README.md)

**Key Topics**:
- Custom LSTM and ConvLSTM implementations
- Action recognition on KTH-Actions dataset
- RNN architectures: LSTMCell, GRUCell, custom implementations
- 3D-CNN (R(2+1)d-Net) for action classification (extra credit)

---

### Assignment 4: Variational Autoencoders (VAEs)

**Location**: `src/Assignment4/`

**Description**: Implementing and training Variational Autoencoders (VAEs) for image generation and reconstruction. Implements both standard Convolutional VAEs (CVAE) and Conditional Convolutional VAEs (CCVAE) on the AFHQ dataset.

**Documentation**: See [Assignment 4 README](https://github.com/Cuda-Vision-Lab/CudaVisionSS2025/blob/main/src/Assignment4/README.md)

**Key Topics**:
- Convolutional VAE (CVAE) implementation
- Conditional Convolutional VAE (CCVAE) with class conditioning
- Reparameterization trick and ELBO optimization
- KL divergence weight experiments
- Latent space visualization and image generation

---

### Assignment 5: Generative Adversarial Networks (GANs)

**Location**: `src/Assignment5/`

**Description**: Implementing and training Generative Adversarial Networks (GANs) for image generation. Implements two variants: DCGAN (Deep Convolutional GAN) for unconditional generation and CDCGAN (Conditional Deep Convolutional GAN) for class-conditional generation.

**Documentation**: See [Assignment 5 README](https://github.com/Cuda-Vision-Lab/CudaVisionSS2025/blob/main/src/Assignment5/README.md)

**Key Topics**:
- Fully convolutional Generator and Discriminator networks
- Adversarial training dynamics
- Unconditional and conditional image generation
- Training stability and monitoring

---

### Assignment 6: Siamese Networks and Contrastive Learning

**Location**: `src/Assignment6/`

**Description**: Implementing and comparing two self-supervised learning approaches for face recognition: TriNet Siamese Networks and SimCLR (Simple Framework for Contrastive Learning). Uses the Labeled Faces in the Wild (LFW) dataset.

**Documentation**: See [Assignment 6 README](https://github.com/Cuda-Vision-Lab/CudaVisionSS2025/blob/main/src/Assignment6/README.md)

**Key Topics**:
- TriNet Siamese model with triplet loss
- SimCLR model with contrastive learning
- ResNet-18 backbone with custom projection heads
- Embedding visualization (PCA, t-SNE)
- Face recognition and clustering

---

### Assignment 7: Vision Transformers for Action Recognition

**Location**: `src/Assignment7/`

**Description**: Implementing and training Vision Transformers (ViT) for action recognition on the KTH-Actions dataset. Explores transformer-based architectures for video classification, comparing different patch sizes and evaluating performance against RNN models.

**Documentation**: See [Assignment 7 README](https://github.com/Cuda-Vision-Lab/CudaVisionSS2025/blob/main/src/Assignment7/README.md)

**Key Topics**:
- Vision Transformer (ViT) implementation
- Patch-based processing for video frames
- Multi-head self-attention mechanisms
- Patch size ablation studies
- Video Vision Transformer (ViViT) with Space-Time attention (extra credit)

---

## Course Project

### Video Prediction with Object Representations

**Location**: `src/CourseProject/`

**Description**: Advanced video prediction using transformer-based architectures. Implements a two-stage pipeline with both holistic and object-centric scene representations on the MOVi-C dataset.

**Documentation**: See [Course Project README](https://github.com/Cuda-Vision-Lab/CudaVisionSS2025/blob/main/src/CourseProject/README.md)

**Key Topics**:
- Two-stage training pipeline (autoencoder + predictor)
- Holistic and object-centric scene representations
- Transformer-based encoders and decoders
- Hybrid CNN + Transformer architecture
- Autoregressive prediction with sliding window mechanism

---

## Project Organization

```
CudaVisionSS2025/
├── src/
│   ├── Assignment1/          # Neural Network Fundamentals
│   ├── Assignment2/          # Transfer Learning and Fine-tuning
│   ├── Assignment3/          # Recurrent Neural Networks
│   ├── Assignment4/          # Variational Autoencoders
│   ├── Assignment5/          # Generative Adversarial Networks
│   ├── Assignment6/          # Self-Supervised Learning
│   ├── Assignment7/          # Vision Transformers
│   └── CourseProject/        # Video Prediction Project
├── docs/                     # Documentation (this site)
├── requirements.txt          # Python dependencies
└── README.md                 # Main repository README
```

Each assignment directory contains:
- **README.md**: Comprehensive documentation for the assignment
- **Notebooks**: Jupyter notebooks with implementations and experiments
- **Source code**: Python modules and utilities
- **Configs**: Experiment configuration files
- **Models**: Saved model checkpoints
- **Logs**: TensorBoard logs and training outputs

---

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone https://github.com/Cuda-Vision-Lab/CudaVisionSS2025.git
   cd CudaVisionSS2025
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Navigate to an assignment**:
   ```bash
   cd src/Assignment1  # or any other assignment
   ```

4. **Read the assignment README**:
   Each assignment directory contains a detailed README.md file with:
   - Overview and objectives
   - Dataset information
   - Model architectures
   - Training instructions
   - Usage examples
   - Results and analysis

---

*For detailed information about each assignment, please refer to the respective README files linked above.*

