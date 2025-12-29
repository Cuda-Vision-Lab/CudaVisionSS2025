# Deep Learning Course Assignments

[![Docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://cuda-vision-lab.github.io/CudaVisionSS2025/)

This repository contains implementations and experiments from a comprehensive deep learning course, covering fundamental neural network architectures, computer vision, generative models, and advanced transformer-based approaches.

## 📚 Overview

This repository is organized into 7 assignments and 1 course project, each focusing on different aspects of deep learning:

- **Assignments 1-2**: Fundamentals of neural networks and transfer learning
- **Assignments 3-4**: Recurrent networks and generative models (VAEs)
- **Assignments 5-6**: Generative adversarial networks and self-supervised learning
- **Assignment 7**: Vision transformers for video understanding
- **Course Project**: Advanced video prediction with transformer architectures

---

***NOTE:*** 

`For detailed information and implementations, please refer to the respective sub-directories.`

---

## 📁 [Assignment 1: Neural Network Fundamentals](src/Assignment1)

**Focus**: Building and training basic neural networks from scratch

- **Dataset**: CIFAR-10 (10-class image classification)
- **Models Implemented**:
  - Multi-Layer Perceptrons (MLPs)
  - Convolutional Neural Networks (CNNs)
- **Key Topics**:
  - Training and validation loops
  - Dropout regularization
  - Custom learning rate schedulers and warmup strategies
  - Model evaluation with confusion matrices
  - Learning curve analysis

---

## 📁 [Assignment 2: Transfer Learning and Fine-tuning](src/Assignment2)

**Focus**: Leveraging pre-trained models for custom classification tasks

- **Task**: Human/Robot binary classification
- **Models Explored**:
  - ResNet18
  - ConvNeXt
  - EfficientNet-B0
- **Approaches Compared**:
  - Full fine-tuning
  - Fixed feature extractor
  - Combined approach (partial fine-tuning)
- **Key Topics**:
  - Transfer learning strategies
  - Data augmentation and normalization
  - Model comparison and evaluation

---

## 📁 [Assignment 3: Recurrent Neural Networks](src/Assignment3)

**Focus**: Implementing RNNs from scratch and applying them to video understanding

- **Task 1**: Implement LSTM and ConvLSTM cells from scratch
- **Task 2**: Action recognition on KTH-Actions dataset
- **Models Implemented**:
  - PyTorch LSTMCell
  - Custom LSTM implementation
  - Custom ConvLSTM implementation
  - GRU cells
- **Task 3**: 3D-CNN (R(2+1)d-Net) for action classification
- **Key Topics**:
  - Recurrent network architectures
  - Temporal sequence modeling
  - Video action recognition
  - 3D convolutions for spatiotemporal features

---

## 📁 [Assignment 4: Variational Autoencoders (VAEs)](src/Assignment4)

**Focus**: Generative models for image reconstruction and conditional generation

- **Models Implemented**:
  - Vanilla VAE (Variational Autoencoder)
  - Convolutional VAE (ConvVAE)
  - Conditional VAE (CVAE)
  - Conditional Convolutional VAE (CCVAE)
- **Dataset**: AFHQ (Animal Faces-High Quality) - cats, dogs, wildlife
- **Key Topics**:
  - Latent space representation learning
  - KL divergence regularization
  - Conditional generation
  - Image reconstruction and generation

---

## 📁 [Assignment 5: Generative Adversarial Networks](src/Assignment5)

**Focus**: Implementing GANs for image generation

- **Models Implemented**:
  - DCGAN (Deep Convolutional GAN)
  - Conditional DCGAN (CDCGAN)
- **Architecture**:
  - Fully convolutional generator and discriminator
  - Conditional generation with class labels
- **Key Topics**:
  - Adversarial training
  - Generator-discriminator dynamics
  - Image generation from noise
  - Comparison with VAE models

---

## 📁 [Assignment 6: Self-Supervised Learning](src/Assignment6)

**Focus**: Learning representations without explicit labels using contrastive learning

- **Task**: Face recognition and embedding learning
- **Dataset**: Labeled Faces in the Wild (LFW)
- **Models Implemented**:
  - TriNet Siamese Network (triplet loss)
  - SimCLR (contrastive learning)
- **Architecture**:
  - ResNet-18 backbone
  - Fully connected embedding layers
  - Normalization layers
- **Key Topics**:
  - Triplet loss and margin optimization
  - Contrastive learning with temperature scaling
  - Embedding visualization (PCA, t-SNE)
  - Face similarity and clustering

---

## 📁 [Assignment 7: Vision Transformers](src/Assignment7)

**Focus**: Applying transformer architectures to video action recognition

- **Task**: Action recognition on KTH-Actions dataset
- **Models Implemented**:
  - Vision Transformer (ViT) with patch-based processing
  - Video Vision Transformer (ViViT) with space-time attention (extra credit)
- **Key Topics**:
  - Image patching and tokenization
  - Multi-head self-attention mechanisms
  - Patch size ablation studies
  - Comparison with RNN-based models from Assignment 3
  - Attention visualization

---

## 📁 [Course Project: Video Prediction with Object Representations](src/CourseProject)

**Focus**: Advanced video prediction using transformer-based architectures

- **Task**: Predict future video frames using learned representations
- **Dataset**: MOVi-C (Multi-Object Video Dataset)
- **Architecture**: Two-stage pipeline
  1. **Autoencoder**: Learn compressed frame representations
  2. **Predictor**: Predict future representations in latent space
- **Approaches**:
  - **Holistic Representation**: Treats entire scene as unified entity
  - **Object-Centric Representation**: Decomposes scenes into individual objects
- **Model Components**:
  - Transformer-based encoders and decoders
  - Hybrid CNN + Transformer architecture for object-centric models
  - Autoregressive prediction with sliding window mechanism
- **Key Features**:
  - Patch-based processing for holistic models
  - Object extraction and composition for OC models
  - Mixed precision training
  - Comprehensive evaluation and visualization

---

## 🛠️ Technical Stack

- **Framework**: PyTorch
- **Visualization**: TensorBoard, Matplotlib
- **Data Processing**: NumPy, PIL, torchvision
- **Evaluation**: sklearn metrics, custom evaluation scripts

---

## 🤝 Contributing
Let's make this project better together!
Contributions are welcome! If you have ideas to improve this project, find a bug, or want to add new features:

- Open an issue to discuss your suggestions or report problems.
- Fork the repository and submit a pull request with your changes.
- Please follow best coding practices and include relevant tests and documentation.

---

## 💬 Support

If you found this project helpful, you can support my work by buying me a coffee or via paypal!  

[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20me%20a%20coffee-orange?logo=buy-me-a-coffee&style=for-the-badge)](https://buymeacoffee.com/amirhosseinsoltani)

[![PayPal](https://img.shields.io/badge/Donate%20with-PayPal-blue?logo=paypal&style=for-the-badge)](https://paypal.me/thesoltanic)






<!-- ## 👥 Team

**Persian CUDA - Deep Learning Team**
- Amirhossein Soltani
- Aidin Masroor
- Moein Taherkhani -->

---

*This repository represents a comprehensive journey through modern deep learning, from basic neural networks to advanced transformer architectures for video understanding.*
