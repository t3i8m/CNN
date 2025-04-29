import numpy as np
from utils.activation_loss import loss_function, sigma_prime_from_a, sigmoid_function, softmax
import random

class NN():
    def __init__(self,*args):
            """Args get all of the layers including input/output and generates weights/biases from the Normal distribution"""
            self.layers = args[0]
            self.num_layers = len(args[0])
            # self.weights = [np.random.randn(curr, prev) for prev, curr in zip(self.layers[0:], self.layers[1:])]

            # self.bias = [np.random.randn(n, 1) for n in self.layers[1:]]

            self.weights     = []
            self.bias        = []

            for prev, curr in zip(self.layers[:-1], self.layers[1:]):
                scale = np.sqrt(1.0 / prev)        
                W = np.random.randn(curr, prev) * scale
                b = np.zeros((curr, 1))           
                self.weights.append(W)
                self.bias.append(b)

            self._dW = [np.zeros_like(w) for w in self.weights]
            self._db = [np.zeros_like(b) for b in self.bias]

    def feedforward(self,y:np.array)->np.array:
        """Calculate the network response on the given input by applying a forward propogation"""
        self.activations = np.empty((self.num_layers,),dtype=object) # 1d array with the length of the layers number
        self.activations[0] = y
        for layer in range(1, self.num_layers-1):
            z = np.dot(self.weights[layer-1], self.activations[layer-1])+self.bias[layer-1] #linear combination of the [num_neuron x prev_layer_neuron] * [prev_layer_neuron x 1] + [num_neuron, 1]
            self.activations[layer] = sigmoid_function(z) # activation function
        z = np.dot(self.weights[-1], self.activations[-2])+self.bias[-1] #linear combination of the [num_neuron x prev_layer_neuron] * [prev_layer_neuron x 1] + [num_neuron, 1]
        self.activations[-1] = softmax(z) # softmax function
        return self.activations[-1]
    
    def backward_from_loss(self, dL_dy: np.ndarray) -> np.ndarray:
        """dL_dy = y_hat − y_true"""
        error = dL_dy
        self._db[-1] += error
        self._dW[-1] += np.dot(error, self.activations[-2].T)

        for l in range(2, self.num_layers):
            a_prev = self.activations[-l - 1]
            a_curr = self.activations[-l]
            error = np.dot(self.weights[-l + 1].T, error) * sigma_prime_from_a(a_curr)
            self._db[-l] += error
            self._dW[-l] += np.dot(error, a_prev.T)

        dL_dx = np.dot(self.weights[0].T, error)
        return dL_dx

    def apply_gradients(self, lr: float, batch_size: int = 1):
        scale = lr / batch_size
        for i in range(len(self.weights)):
            self.weights[i] -= scale * self._dW[i]
            self.bias[i] -= scale * self._db[i]
            self._dW[i].fill(0)
            self._db[i].fill(0)

    def state_dict(self) -> dict:
        return {
            "weights": self.weights,   
            "biases":  self.bias
        }
    
    def set_state_dict(self, state: dict) -> None:
        self.weights = [w.copy() for w in state["weights"]]
        self.bias    = [b.copy() for b in state["biases"]]
