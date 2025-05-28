# Convolutional Neural Network for MNIST Digit Recognition

![image](https://github.com/user-attachments/assets/215e95ff-b070-4a86-8a39-5f1d0f2ebe40)
![Source: Medium article by Abhishek Jain]([[https://miro.medium.com/v2/resize:fit:1400/format:webp/1*fD1kDcF3K4C9Mr6Niz2pNg.png](https://medium.com/@abhishekjainindore24/understanding-convolutional-neural-networks-cnns-with-an-example-on-the-mnist-dataset-a64815843685)](https://medium.com/@abhishekjainindore24/understanding-convolutional-neural-networks-cnns-with-an-example-on-the-mnist-dataset-a64815843685))

This project implements a Convolutional Neural Network (CNN) from scratch (without Keras/TensorFlow/PyTorch) to classify handwritten digits from the MNIST dataset.

## 🧠 Project Goal
To better understand **how convolutional neural networks work under the hood** by building one from scratch without high-level ML frameworks.

## 📁 Project Structure
```
CNN/
├── checkpoints/ # Saved model weights 
├── data/
│ └── mnist.pkl.gz # Compressed MNIST data file
├── model/
│ ├── CNN.py # Main CNN architecture
│ ├── ConvLayer.py # Custom convolutional layer
│ ├── init.py
│ └── FFNN/
│ └── NN.py # Fully-connected network 
├── utils/
│ ├── activation_loss.py # Activation functions and loss functions
│ ├── mnist_loader.py # MNIST data loader
│ ├── init.py
├── main.py # Entry point for training and evaluation
├── training.py # Training loop
```
## 📦 Requirements
- Python 3.7+
- NumPy
- gzip, pickle (from Python standard library)

To install the required packages: ```pip install numpy```

## 🚀 How to Run
1. Make sure `mnist.pkl.gz` is present in the `data/` folder.
2. Run training: ```python main.py```

## ✅ Features
- Custom implementation of convolutional layers
- Custom implementation for both CNN and FFNN architectures
- MNIST digit classification
- Modular and extensible codebase

> ⚠️ Training and hyperparameter tuning are **still in progress**. Results shown are preliminary.

