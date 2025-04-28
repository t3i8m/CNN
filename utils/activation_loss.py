import numpy as np

def sigmoid_function(x:np.array)->np.array:
    """A sigmoid activation function implementation"""
    return 1.0/(1.0+ np.exp(-x))

def loss_function(predicted:np.array, real:np.array):
    """Loss function to calculate the sum of squared errors (SSE)"""
    return np.sum((predicted-real)**2)

def sigma_prime_from_a(a:np.array)->np.array:
    """Sigma prime from the activations"""
    return a * (1 - a)

def softmax(z:np.array):
    exp_z = np.exp(z - np.max(z))
    return exp_z / np.sum(exp_z)

