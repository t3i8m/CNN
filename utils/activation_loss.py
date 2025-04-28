import numpy as np
SIGMOID_CLIP = 709.0 

def sigmoid_function(x:np.array)->np.array:
    """A sigmoid activation function implementation"""
    x = np.clip(x, -SIGMOID_CLIP, SIGMOID_CLIP)
    return 1.0 / (1.0 + np.exp(-x))

def loss_function(predicted:np.array, real:np.array):
    """Loss function to calculate the sum of squared errors (SSE)"""
    return np.sum((predicted-real)**2)

def cross_entropy(y_hat: np.ndarray, y_true: np.ndarray) -> float:
    """
    y_hat : shape (batch, C), softmax (>0, <=1, sum=1)
    y_true: shape (batch, C), one-hot
    """
    eps = 1e-12 #!=0                             
    return -np.mean(np.sum(y_true * np.log(y_hat + eps), axis=1))


def sigma_prime_from_a(a:np.array)->np.array:
    """Sigma prime from the activations"""
    return a * (1 - a)

def softmax(z:np.array):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

