import numpy as np
from .ConvLayer import ConvLayer


class CNN():

    def __init__(self, *args):
        """Initializes convolutional neural network with dynamic number of arguments:
        - *args - [[kernel_size_1st_layer, number_channels_1st_layer], [kernel_size_2nd_layer, number_channels_2nd_layer]...]"""

        # we check the input data size and data types
        self.conv_layers = []
        for layer in args[0]:
            if not isinstance(layer, (list, list)):
                raise ValueError("Each layer must be a list or tuple, like [kernel_size, number_channels]")

            if len(layer) != 2:
                raise ValueError("Each layer description must have exactly two elements: [kernel_size, number_channels]")

            kernel_size, number_channels = layer
            if not isinstance(kernel_size, int) or not isinstance(number_channels, int):
                raise TypeError("Kernel size and number of channels must be integers")

            if kernel_size <= 0 or number_channels <= 0:
                raise ValueError("Kernel size and number of channels must be positive integers")
            
            conv_layer = ConvLayer(kernel_size=layer[0], out_channels=layer[1])
            self.conv_layers.append(conv_layer)

        
    def feed_forward(self, y:np.array)->np.array:
        """Calculate the network response on the given input by applying a forward propogation
            - y - (nxn) input image encoded matrix"""

        for layer in self.conv_layers:
            
            feature_maps = 
        



