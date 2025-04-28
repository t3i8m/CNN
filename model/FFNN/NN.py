import numpy as np
from utils.activation_loss import loss_function, sigma_prime_from_a, sigmoid_function, softmax
import random

class NN():
    def __init__(self,*args):
            """Args get all of the layers including input/output and generates weights/biases from the Normal distribution"""
            self.layers = args[0]
            self.num_layers = len(args[0])
            self.weights = [np.random.randn(curr, prev) for prev, curr in zip(self.layers[0:], self.layers[1:])]

            self.bias = [np.random.randn(n, 1) for n in self.layers[1:]]

    def feedforward(self,y:np.array)->np.array:
        """Calculate the network response on the given input by applying a forward propogation"""
        self.activations = np.empty((self.num_layers,),dtype=object) # 1d array with the length of the layers number
        self.activations[0] = y
        for layer in range(1, self.num_layers-1):
            z = np.dot(self.weights[layer-1], self.activations[layer-1])+self.bias[layer-1] #linear combination of the [num_neuron x prev_layer_neuron] * [prev_layer_neuron x 1] + [num_neuron, 1]
            self.activations[layer] = sigmoid_function(z) # activation function
        z = np.dot(self.weights[-2], self.activations[-2])+self.bias[-2] #linear combination of the [num_neuron x prev_layer_neuron] * [prev_layer_neuron x 1] + [num_neuron, 1]
        self.activations[-1] = softmax(z) # softmax function
        return self.activations[-1]
    
    def SGD(self, training_data:list, epochs:int, mini_batch_size:int, learning_rate:float, test_data = None)->None:
        """Stochastic gradient descent algorithm to train the network"""
        if(test_data):
            test_size = len(test_data)

        train_size = len(training_data)

        # each epoch we go through all of the training data and create mini_batches
        for epoch in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[n:n+mini_batch_size] for n in range(0, train_size, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, learning_rate)

            if(test_data):
                print(f"Epoch: {epoch} completed. Result {self.evaluate(test_data)}/{test_size} ")
            else:
                print(f"Epoch {epoch} completed")
        return
    
    def update_mini_batch(self, mini_batch:list, learning_rate:float)->None:
        """Go through all of the batch and use backprop to compute gradients, compute the average"""
        gradients_holder_weights = [np.zeros_like(w) for w in self.weights]
        gradients_holder_bias = [np.zeros_like(b) for b in self.bias]

        for index, batch in enumerate(mini_batch):
            gradients = self.backprop(batch)

            for n in range(len(gradients[0])):
                gradients_holder_weights[n]+=gradients[0][n]
                gradients_holder_bias[n]+=(gradients[1][n])
        
        self.weights = [self.weights[w]-(gradients_holder_weights[w]*(1/len(mini_batch))*learning_rate) for w in range(len(self.weights))]
        self.bias = [self.bias[b]-(gradients_holder_bias[b]*(1/len(mini_batch))*learning_rate) for b in range(len(self.bias))]

        return

    def backprop(self, batch:list)->tuple:
        """Use a backpropogation algorithm to calculate gradients for each weight and bias."""

        gr_weights = [np.zeros_like(w) for w in self.weights]
        gr_bias = [np.zeros_like(w) for w in self.bias]
        self.feedforward(batch[0])

        activations = self.activations
        error = (activations[-1]-batch[1])*sigma_prime_from_a(activations[-1]) # error has shape of the vector [neuron x 1]
        gr_bias[-1] = error
        gr_weights[-1] = np.dot(error, activations[-2].T) # gradients for the output weights with the shape of [neuron x prev_layer_neuron]
        
        for layer in range(2, self.num_layers):
            error = np.dot(self.weights[-layer+1].T, error)*sigma_prime_from_a(activations[-layer])  # error has shape of the vector [neuron x 1]
            gr_weights[-layer] = np.dot(error, activations[-layer-1].T) # gradients shape [neuron x prev_layer_neuron]
            gr_bias[-layer]= error
        return (gr_weights, gr_bias)
    
    def evaluate(self, test_data:list)->int:
        """Use a feedforward to check the number of correct guesses with true labels"""
        counter = 0
        for batch in test_data:
            output = self.feedforward(batch[0])
            predicted = np.argmax(output)
            if(predicted==batch[1]):
                counter+=1

        return counter