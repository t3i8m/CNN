import numpy as np


class ConvLayer():

    def __init__(self, kernel_size:int, out_channels:int, pooling=2):
        """Initializer for the convolution layer:
            - kernel_size - (height x width) of the filter (e.g. 3x3),
            - out_channels - number of features(filters) we extract and get feature maps"""
        
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.filters = [np.random.randn(kernel_size, kernel_size) for _ in range(out_channels)]
        self.biases = [np.random.randn() for  _ in range(out_channels)]
        self.feature_maps = []
        self.max_poolings = []
        self.pooling = pooling
        
    def forward(self, source:np.array)->(np.array):
        """Apply forward propogation for the single conv layer"""
        self.feature_maps = []
        for index, fltr in enumerate(self.filters):
            for src in source:
                fm = self.extract_feature_map(fltr, src, index)
                self.feature_maps.append(fm)
                self.max_poolings.append(self.max_pooling(fm, self.pooling ))
        return self.feature_maps
    
    def extract_feature_map(self, filter:np.array, src:np.array, filter_id:int)->(np.array):
        """Apply a single filter over the input src and return the feature map."""
        h_src, w_src = src.shape
        h_filter, w_filter = filter.shape

        h_out = h_src - h_filter + 1
        w_out = w_src - w_filter + 1

        feature_map = np.zeros((h_out, w_out))

        for y in range(h_out):
            for x in range(w_out):
                cut = src[y:y+h_filter, x:x+w_filter]

                dot_product = np.sum(cut * filter)+self.biases[filter_id]
                # ReLu
                feature_map[y, x] = max(0, dot_product)

        return feature_map      

    def max_pooling(self, feature_map:np.array, pool_size:int):
        """Apply a max pooling filter over the input src and return the new feature map."""

        h_src, w_src = feature_map.shape

        # stride = pool_size
        h_out = h_src // pool_size
        w_out = w_src // pool_size

        pooled_map = np.zeros((h_out, w_out))
        for y in range(h_out):
            for x in range(w_out):
                cut = feature_map[y:y+pool_size, x:x+pool_size]

                max_elm = np.max(cut)
                pooled_map[y, x] = max_elm
        return pooled_map