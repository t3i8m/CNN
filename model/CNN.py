import numpy as np
from .ConvLayer import ConvLayer


class CNN():

    def __init__(self, layers, first_in_channels: int = 1):
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

        
    def feed_forward(self, y:np.array)->np.array:
        """Calculate the network response on the given input by applying a forward propogation
            - y - (nxn) input image encoded matrix"""

        outputs = [y]           
        for layer in self.conv_layers:
            outputs = layer.forward(outputs)

        # transfrom to a (1xn) vector
        flat = []
        for feature_map in outputs:
            flat.extend(feature_map.flatten())

        return flat

