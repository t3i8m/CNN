import numpy as np


class ConvLayer():

    def __init__(self, kernel_size:int, out_channels:int, in_channels:int, pooling=2):
        """Initializer for the convolution layer:
            - kernel_size - (height x width) of the filter (e.g. 3x3),
            - out_channels - number of features(filters) we extract and get feature maps"""
        
        self.kernel_size = kernel_size
        self.out_channels = out_channels

        self.filters = [[np.random.randn(kernel_size, kernel_size) for _ in range(in_channels)] for _ in range(out_channels)]
        self.biases = [np.random.randn() for  _ in range(out_channels)]
        self.feature_maps = []
        self.max_poolings = []
        self.pooling = pooling

        # gradients
        self._dW = [[np.zeros_like(f) for f in filt_group] for filt_group in self.filters]
        self._db = [0.0 for _ in range(out_channels)]

        #cache for the backprop
        self._inputs = None
        self._feature_maps = None
        self._pool_indices = None  
        
    def forward(self, sources):
        self._inputs = sources  
        self._feature_maps = []
        pooled_out = []
        self._pool_indices = []

        self.feature_maps = []
        self.max_poolings = []

        for filt_id, filt_group in enumerate(self.filters):
            summed_fm = None
            for kernel, src in zip(filt_group, sources):
                fm = self.convolve(kernel, src, filt_id)
                summed_fm = fm if summed_fm is None else summed_fm + fm

            summed_fm += self.biases[filt_id]
            relu_map = np.maximum(summed_fm, 0) # ReLu activation
            pooled_map, idx = self.max_pooling(relu_map, self.pooling)
            self._feature_maps.append(relu_map)
            pooled_out.append(pooled_map)
            self._pool_indices.append(idx)
        return pooled_out
    
    def backward(self, d_pool:np.ndarray) -> (np.ndarray):
        """Apply gradient to the max-pool and return gradient to the prev layer"""

        d_prev= [np.zeros_like(src) for src in self._inputs]
        k = self.kernel_size

        for oc in range(self.out_channels):
            grad_pooled = d_pool[oc]                      # (h_out, w_out)
            relu_map = self._feature_maps[oc]             # (h_conv, w_conv)
            idx_map = self._pool_indices[oc]              # (h_out, w_out, 2)

            d_relu = np.zeros_like(relu_map)
            for (y, x), grad_val in np.ndenumerate(grad_pooled):
                iy, ix = idx_map[y, x]
                if relu_map[iy, ix] > 0:                  # ReLU
                    d_relu[iy, ix] += grad_val

            # gradient for the bias
            self._db[oc] += d_relu.sum()

            for ic, src in enumerate(self._inputs):
                # gradients
                dW = self._dW[oc][ic]
                h_out, w_out = d_relu.shape
                for ky in range(k):
                    for kx in range(k):
                        window = src[ky:ky + h_out, kx:kx + w_out]
                        dW[ky, kx] += np.sum(window * d_relu)

                # to the prev layer
                rot_filt = np.flip(self.filters[oc][ic])
                pad_h, pad_w = k - 1, k - 1
                padded = np.pad(d_relu, ((pad_h, pad_h), (pad_w, pad_w)))
                dX = d_prev[ic]
                for y in range(src.shape[0]):
                    for x in range(src.shape[1]):
                        region = padded[y:y + k, x:x + k]
                        dX[y, x] += np.sum(region * rot_filt)

        return d_prev  

    def apply_gradients(self, lr: float, batch_size: int = 1) -> None:
        """AAApply the gradients"""
        scale = lr / batch_size
        for oc in range(self.out_channels):
            for ic in range(self.in_channels):
                self.filters[oc][ic] -= scale * self._dW[oc][ic]
                self._dW[oc][ic].fill(0)
            self.biases[oc] -= scale * self._db[oc]
            self._db[oc] = 0.0
    
    def convolve(self, filter:np.array, src:np.array, filter_id:int)->(np.array):
        """Apply a single filter over the input src and return the feature map."""
        h_src, w_src = src.shape
        h_filter, w_filter = filter.shape

        h_out = h_src - h_filter + 1
        w_out = w_src - w_filter + 1

        feature_map = np.zeros((h_out, w_out))

        for y in range(h_out):
            for x in range(w_out):
                cut = src[y:y+h_filter, x:x+w_filter]

                summed = np.sum(cut * filter)
                # ReLu
                feature_map[y, x] = max(0, summed)

        return feature_map      
    


    def max_pooling(self, feature_map:np.array, pool_size:int):
        """Apply a max pooling filter over the input src and return the new feature map."""

        h_src, w_src = feature_map.shape

        # stride = pool_size
        h_out = h_src // pool_size
        w_out = w_src // pool_size

        pooled_map = np.zeros((h_out, w_out))
        indices = np.zeros((h_out, w_out, 2), dtype=int)
        for y in range(h_out):
            for x in range(w_out):
                cut = feature_map[y:y+pool_size, x:x+pool_size]
                y0, x0 = y * pool_size, x * pool_size

                local_idx = np.unravel_index(np.argmax(cut), cut.shape)

                iy, ix = y0 + local_idx[0], x0 + local_idx[1]
                max_elm = np.max(cut)
                pooled_map[y, x] = max_elm
                indices[y, x] = (iy, ix)
        return pooled_map, indices
    
    def state_dict(self) -> dict:
        return {
            "filters": self.filters,   
            "biases":  self.biases
        }

    def set_state_dict(self, state: dict) -> None:
        """
        state = {
            "filters": list[list[np.ndarray]],
            "biases" : list[np.ndarray  или  float]
        }
        """
        self.filters = [ [f.copy() for f in group] for group in state["filters"] ]
        self.biases  = [ b.copy() if isinstance(b, np.ndarray) else b
                         for b in state["biases"] ]