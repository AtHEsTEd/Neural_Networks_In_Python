import numpy as np
from numpy import ndarray
from scipy.signal import correlate2d, convolve2d


class LayerDense:

    def __init__(self, n_inputs: int, n_neurons: int, weights_multiple: float = 0.01,
                 weight_regularizer_l1: float = 0,
                 bias_regularizer_l1: float = 0, weight_regularizer_l2: float = 0, bias_regularizer_l2: float = 0):
        """
        Initialises random weights
        Creates biases as 2d NumPy array of 0s
        Sets activation function object
        :param n_inputs: How many inputs the layer has
        :param n_neurons: How many neurones are in the layer
        :param weights_multiple: What value the random weights are multiplied by to normalise them (default: 0.01)
        :param weight_regularizer_l1: Lambda value for l1 regularization for weights
        :param bias_regularizer_l1: Lambda value for l1 regularization for biases
        :param weight_regularizer_l2: Lambda value for l2 regularization for weights
        :param bias_regularizer_l2: Lambda value for l2 regularization for biases
        """
        self.weights = weights_multiple * np.random.randn(n_inputs, n_neurons)
        self.biases = np.random.randn(1, n_neurons)
        self.weight_regularizer_l1 = weight_regularizer_l1
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.weight_regularizer_l2 = weight_regularizer_l2
        self.bias_regularizer_l2 = bias_regularizer_l2

    def forward(self, inputs: np.ndarray, training: bool) -> None:
        """
        Performs forward pass using matrix multiplication
        :param inputs: Input matrix
        :return: None. Output matrix stored in self.output
        """
        self.inputs = inputs
        output = np.dot(inputs, self.weights)
        output = output + self.biases
        self.output = output

    def backward(self, dvalues: ndarray) -> None:
        """
        Performs backward pass
        :param dvalues: Values
        :return: None, values stored in self.dweights, self.dbiases, self.dinputs
        """
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)

        # Regularization gradients
        # L1 - Weights
        if self.weight_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regularizer_l1 * dL1

        # L2 - Weights
        if self.weight_regularizer_l2 > 0:
            self.dweights += 2 * self.weight_regularizer_l2 * self.weights

        # L1 - Biases
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dbiases += self.bias_regularizer_l1 * dL1

        # L2 - Biases
        if self.bias_regularizer_l2 > 0:
            self.dbiases += 2 * self.bias_regularizer_l2 * self.biases

        self.dinputs = np.dot(dvalues, self.weights.T)

    def get_parameters(self) -> tuple[ndarray]:
        # Retrieve layer parameters
        return self.weights, self.biases

    def set_paramters(self, weights: ndarray, biases: ndarray):
        self.weights = weights
        self.biases = biases


class Dropout:

    def __init__(self, rate: float):
        """
        Dropout layers disable a certain amount of neurones each forward pass to prevent overfitting and reliance
        on certain neurones
        :param rate: Rate that neurones are turned off, between 0 and 1
        """
        if rate > 1 or rate < 0:
            raise ValueError("Rate must be between 0 and 1")
        self.rate = 1 - rate

    def forward(self, inputs: ndarray, training: bool) -> None:
        # save inputs
        self.inputs = inputs

        # If not in training, dont engage dropout
        if not training:
            self.output = inputs.copy()
            return

        # generate and save mask
        self.binary_mask = np.random.binomial(1, self.rate, size=inputs.shape) / self.rate

        # apply to inputs to generate outputs
        self.output = inputs * self.binary_mask

    def backward(self, dvalues: ndarray) -> None:
        self.dinputs = dvalues * self.binary_mask


class Input:

    def forward(self, inputs, training: bool) -> None:
        self.output = inputs


class Flatten:

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.shape = (self.height, self.width)

    def forward(self, inputs: ndarray, training: bool) -> None:
        # self.shape = inputs.shape
        # self.output = inputs.reshape(inputs.shape[0], -1)
        self.input_shape = inputs.shape
        self.output = inputs.reshape(*self.shape)

    def backward(self, dvalues: ndarray) -> None:
        self.dinputs = dvalues.reshape(*self.input_shape)
        return

    class Conv2D:

        def __init__(self, input_shape: tuple[int, int, int], kernel_size: int, depth: int):
            input_depth, input_height, input_width = input_shape
            self.depth = depth
            self.input_shape = input_shape
            self.input_depth = input_depth
            self.output_shape = (depth * input_depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
            self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
            self.weights = np.random.randn(*self.kernels_shape)
            self.biases = np.random.randn(*self.output_shape)

        def forward(self, input: ndarray, training: bool):
            self.input = input
            self.output = np.copy(self.biases)

            for i in range(self.depth):
                for j in range(self.input_depth):
                    self.output[i] += correlate2d(self.input[j], self.weights[i, j], mode="valid")

        def backward(self, dvalues: ndarray):
            self.dweights = np.zeros(self.kernels_shape)
            self.dinputs = np.zeros(self.input_shape)

            for i in range(self.depth):
                for j in range(self.input_depth):
                    self.dweights[i, j] = correlate2d(self.input[j], dvalues[i], mode="valid")
                    self.dinputs[j] += convolve2d(dvalues[i], self.weights[i, j], mode="full")

            self.dbiases = np.copy(dvalues)

    class MaxPooling:

        def __init__(self, pool_size: int):
            self.pool_size = pool_size

        def forward(self, input: ndarray, training: bool) -> None:
            self.input = input
            self.num_channels, self.input_height, self.input_width = input.shape
            self.output_height = self.input_height // self.pool_size
            self.output_width = self.input_width // self.pool_size

            self.output = np.zeros((self.num_channels, self.output_height, self.output_width))

            # Iterate over channels
            for c in range(self.num_channels):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        start_i = i * self.pool_size
                        start_j = j * self.pool_size

                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size

                        patch = input[c, start_i:end_i, start_j:end_j]
                        self.output[c, i, j] = np.max(patch)

        def backward(self, dvalues: ndarray):
            self.dinputs = np.zeros_like(self.input)

            # No gradient calculations, just take the max of the gradients
            nums = self.dinputs.shape[0]
            for c in range(self.num_channels):
                for i in range(self.output_height):
                    for j in range(self.output_width):
                        start_i = i * self.pool_size
                        start_j = j * self.pool_size

                        end_i = start_i + self.pool_size
                        end_j = start_j + self.pool_size
                        patch = self.input[c, start_i:end_i, start_j:end_j]

                        mask = patch == np.max(patch)
                        self.dinputs[c, start_i:end_i, start_j:end_j] = dvalues[c, i, j] * mask
