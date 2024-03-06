import numpy as np
from numpy import ndarray
import Layers
import ActivationFunctions


class Optimizer:

    def __init__(self, learning_rate: float, decay: float):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0

    def decay_pre_update(self) -> None:
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iterations))

    def update_params(self,
                      layer: Layers.LayerDense | ActivationFunctions.ActivationFunction | 
                             ActivationFunctions.ActivationFunctions.ActivationLoss.ActLoss):
        """
        Updates the parameters for a layer inplace
        :param layer: Layer to update parameters for
        """
        pass

    def post_update(self) -> None:
        self.iterations += 1


class SGD(Optimizer):

    def __init__(self, learning_rate: float = 1, decay: float = 0, momentum: float = 0):
        """
        Initialises a Stochastic Gradient Descent object
        :param learning_rate: Beginning value that the gradients are multiplied by
        :param decay: Value that determines how quickly the learning rate decreases throughout training
        :param momentum: What percentage of the previous weights are taken into account when training
        """
        # self.learning_rate = learning_rate
        # self.current_learning_rate = learning_rate
        # self.decay = decay
        # self.iterations = 0
        super().__init__(learning_rate, decay)
        self.momentum = momentum

    def update_params(self,
                      layer: Layers.LayerDense | ActivationFunctions.ActivationFunction | ActivationFunctions.ActivationFunctions.ActivationLoss.ActLoss):
        # if momentum is used
        if self.momentum:

            # if layer doesn't have momentum arrays, create them
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            # Update weights with moments
            # Take previous updates multiplied by momentum and update with current gradients
            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = weight_updates

            # do the same for biases
            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
            layer.bias_momentums = bias_updates

        # Normal SGD updates without momentum
        else:
            weight_updates = -self.current_learning_rate * layer.dweights
            bias_updates = -self.current_learning_rate * layer.dbiases

        # Update weights and biases
        layer.weights += weight_updates
        layer.biases += bias_updates


class Adagrad(Optimizer):

    def __init__(self, learning_rate: float = 1, decay: float = 0, epsilon: float = 1e-7):
        """
        Initialises an Adaptive Gradient Descent object
        :param learning_rate: Beginning value that the gradients are multiplied by
        :param decay: Value that determines how quickly the learning rate decreases throughout training
        :param epsilon: Small value to prevent division by 0 when updating parameters
        """
        # self.learning_rate = learning_rate
        # self.current_learning_rate = learning_rate
        # self.decay = decay
        # self.iterations = 0
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon

    def update_params(self,
                      layer: Layers.LayerDense | ActivationFunctions.ActivationFunction | ActivationFunctions.ActivationLoss.ActLoss):
        # If layer does not contain cache arrays create them filled with 0s
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update cache with squared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # SGD Parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)


class RMSprop(Optimizer):

    def __init__(self, learning_rate: float = 0.001, decay: float = 0, epsilon: float = 1e-7, rho: float = 0.9):
        """
        Initialises an Adaptive Gradient Descent object
        :param learning_rate: Beginning value that the gradients are multiplied by
        :param decay: Value that determines how quickly the learning rate decreases throughout training
        :param epsilon: Small value to prevent division by 0 when updating parameters
        :param rho: Cache memory decay rate, the rate at which squared gradients are added to the cache
        """
        # self.learning_rate = learning_rate
        # self.current_learning_rate = learning_rate
        # self.decay = decay
        # self.iterations = 0
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.rho = rho

    def update_params(self,
                      layer: Layers.LayerDense | ActivationFunctions.ActivationFunction | ActivationFunctions.ActivationLoss.ActLoss):
        # If layer does not contain cache arrays create them filled with 0s
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update cache with squared current gradients using rho to carry over momentum
        layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases ** 2

        # SGD Parameter update + normalization with square rooted cache
        layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)


class Adam(Optimizer):

    def __init__(self, learning_rate: float = 0.001, decay: float = 0, epsilon: float = 1e-7,
                 beta_1: float = 0.9, beta_2: float = 0.999):
        """
        Initialises an Adaptive Gradient Descent object
        :param learning_rate: Beginning value that the gradients are multiplied by
        :param decay: Value that determines how quickly the learning rate decreases throughout training
        :param epsilon: Small value to prevent division by 0 when updating parameters
        :param beta_1: Momentums and caches are divided by (1 - beta_1 ** iteration) to speed up training
        :param beta_2: Cache memory decay rate, the rate at which squared gradients are added to the cache
        """
        # self.learning_rate = learning_rate
        # self.current_learning_rate = learning_rate
        # self.decay = decay
        # self.iterations = 0
        super().__init__(learning_rate, decay)
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self,
                      layer: Layers.LayerDense | ActivationFunctions.ActivationFunction | ActivationFunctions.ActivationLoss.ActLoss):
        # If layer does not contain cache arrays create them filled with 0s
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # Update momentum with current gradients
        layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
        layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases

        # Get corrected momentums
        # Iterations is at 0 to start to add 1
        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))

        # Update cache with squared current gradients
        layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights ** 2
        layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.dbiases ** 2

        # get corrected cache
        weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))

        # SGD parameter update + normalisation with square rooted cache
        layer.weights += (-self.current_learning_rate * weight_momentums_corrected /
                          (np.sqrt(weight_cache_corrected) + self.epsilon))
        layer.biases += (-self.current_learning_rate * bias_momentums_corrected /
                         (np.sqrt(bias_cache_corrected) + self.epsilon))
