import numpy as np
from .ConvLayer import ConvLayer
import random
from .FFNN.NN import NN
from utils.activation_loss import cross_entropy, sigma_prime_from_a, sigmoid_function, softmax
import os, pickle
import matplotlib.pyplot as plt

class CNN():

    def __init__(self, layers, first_in_channels: int = 1, output_vector = 400):
        """Initializes convolutional neural network with dynamic number of arguments:
        - layers - [[kernel_size_1st_layer, number_channels_1st_layer], [kernel_size_2nd_layer, number_channels_2nd_layer]...]"""

        # we check the input data size and data types
        self.conv_layers = []
        prev_channels = first_in_channels
        for layer in layers:

            if len(layer) != 2:
                raise ValueError("Each layer description must have exactly two elements: [kernel_size, number_channels]")

            kernel_size, number_channels = layer
            if not isinstance(kernel_size, int) or not isinstance(number_channels, int):
                raise TypeError("Kernel size and number of channels must be integers")

            if kernel_size <= 0 or number_channels <= 0:
                raise ValueError("Kernel size and number of channels must be positive integers")
            
            conv_layer = ConvLayer(kernel_size=layer[0], out_channels=layer[1], in_channels=prev_channels)
            prev_channels = number_channels
            self.conv_layers.append(conv_layer)
            self.cnn_output = output_vector
            self.ffnn = None

        
    def feed_forward(self, y:np.array)->np.array:
        """Calculate the network response on the given input by applying a forward propogation
            - y - (nxn) input image encoded matrix
            Output: one high dimensional vector"""

        maps = [y]           
        for layer in self.conv_layers:
            maps = layer.forward(maps)

        self._cached_maps = maps

        # transfrom to a (1xn) vector
        flat = self._flatten(maps)
        if self.ffnn is None:
            self.ffnn = NN([flat.shape[0], 30, 10])
        # print("flat: ", len(flat))
        predictions = self.ffnn.feedforward(flat)

        return predictions

    def SGD(self, training_data:list, epochs:int, mini_batch_size:int, learning_rate:float, test_data = None)->None:
        """Stochastic gradient descent algorithm to train the network"""
        if(test_data):
            test_size = len(test_data)

        train_size = len(training_data)

        for epoch in range(epochs):
            loss_history = []


            random.shuffle(training_data)
            print(f"Entering Epoch #{epoch+1}")
            mini_batches = [training_data[n:n+mini_batch_size] for n in range(0, train_size, mini_batch_size)]
            epoch_loss = 0.0
            for mini_batch in mini_batches:
                xs = [x for x, _ in mini_batch]
                ys = [self._one_hot(lbl) for _, lbl in mini_batch]

                for x, y_true in zip(xs, ys):
                    y_hat = self.feed_forward(x)
                    loss = cross_entropy(y_hat, y_true)
                    epoch_loss += loss

                    dL_dy = y_hat - y_true # gradients by the output

                    # for layer in self.conv_layers:
                    #     x = layer.forward([x])[0]
                    #     self._cached_maps.append(x)

                    self.backward(dL_dy)

                for layer in self.conv_layers:
                    layer.apply_gradients(learning_rate/mini_batch_size)  
                self.ffnn.apply_gradients(learning_rate/mini_batch_size)   

            avg_loss = epoch_loss / len(training_data)
            print(f"Epoch {epoch + 1}: loss = {avg_loss:.4f}")
            loss_history.append(avg_loss)  

            if test_data:
                correct = 0
                for x, lbl in test_data:
                    pred = np.argmax(self.feed_forward(x))
                    if pred == lbl:
                        correct += 1
                acc = correct / len(test_data)
                print(f"           val_acc = {acc:.3%}")

        self.save_weights("checkpoints/epoch10.npz")  
        plt.plot(range(1, epochs + 1), loss_history, marker='o')
        plt.title("Training Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.grid(True)
        plt.show()

        return
    

    def backward(self, dL_dy: np.ndarray) -> None:
        """Call back‑prop FFNN and ConvLayers, store delta."""

        # FFNN: returns delatW/deltab and gradients by (dL/dflat)
        dL_dflat = self.ffnn.backward_from_loss(dL_dy)  # shape (flat_dim, 1)

        last_maps_shapes = [fm.shape for fm in self._cached_maps]
        cursor = 0
        grads_to_maps = []
        for shape in last_maps_shapes:
            size = np.prod(shape)
            grads_to_maps.append(dL_dflat[cursor:cursor + size].reshape(shape))
            cursor += size

        d_prev = grads_to_maps  # gradients for the last conv layer

        for layer in reversed(self.conv_layers):
            d_prev = layer.backward(d_prev)

    def state_dict(self) -> dict:
        return {
            "conv":  [layer.state_dict() for layer in self.conv_layers],
            "ffnn":  self.ffnn.state_dict()
        }
    
    def save_weights(self, path: str, *, compress: bool = True) -> None:
        """Save weights/biases into the file"""
        state = self.state_dict()

        if path.endswith(".npz"):
            flat = {}
            
            for i, layer in enumerate(state["conv"]):
                for j, f in enumerate(layer["filters"]):
                    flat[f"conv{i}_W{j}"] = np.stack(f)    
                flat[f"conv{i}_b"] = np.asarray(layer["biases"])

            for k, W in enumerate(state["ffnn"]["weights"]):
                flat[f"ff_W{k}"] = W
                flat[f"ff_b{k}"] = state["ffnn"]["biases"][k]

            if compress:
                np.savez_compressed(path, **flat)
            else:
                np.savez(path, **flat)

        else: 
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(state, f)

    def load_weights(self, path: str) -> None:
        if path.endswith(".npz"):
            data = np.load(path)
            state = {"conv": [], "ffnn": {"weights": [], "biases": []}}

            n_conv = len(self.conv_layers)
            for i in range(n_conv):
                filters = []
                j = 0
                while f"conv{i}_W{j}" in data.files:
                    filters.append([f for f in data[f"conv{i}_W{j}"]])
                    j += 1
                biases = data[f"conv{i}_b"]
                state["conv"].append({"filters": filters, "biases": biases})

            k = 0
            while f"ff_W{k}" in data.files:
                state["ffnn"]["weights"].append(data[f"ff_W{k}"])
                state["ffnn"]["biases"].append(data[f"ff_b{k}"])
                k += 1

        else:  # pickle
            with open(path, "rb") as f:
                state = pickle.load(f)

        self.load_state_dict(state)

    @staticmethod
    def _flatten(feature_maps: np.ndarray) -> np.ndarray:
        """Concat all of the feature maps into a column (n, 1)."""
        return np.concatenate([fm.flatten() for fm in feature_maps])[:, None]

    def _one_hot(self, label, num_classes: int = 10) -> np.ndarray:
        
        if isinstance(label, np.ndarray) and label.size == num_classes and label.max() == 1:
            return label.reshape(num_classes, 1)

        if isinstance(label, np.ndarray):
            label = int(label.flatten()[0])

        y = np.zeros((num_classes, 1))
        y[int(label)] = 1.0
        return y

    
