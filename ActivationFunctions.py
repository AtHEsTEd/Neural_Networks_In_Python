import numpy as np
from numpy import ndarray
import Loss


class ActivationFunction:

    def forward(self, x: ndarray | float | int, training: bool) -> None:
        """
        :param x: Value to be passed through activation function
        :return: Forward pass of the activation function stored in output
        """
        pass

    def backward(self, dvalues: ndarray) -> None:
        """
        Derivative of forward pass
        :param dvalues: values to pass through the derivative
        :return: None, stored in dinputs
        """
        pass

    def predictions(self, outputs: ndarray) -> ndarray:
        """
        Returns predictions of model
        :param outputs: Outputs to calculate predictions from
        :return: The predictions
        """
        pass


class ReLU(ActivationFunction):

    def forward(self, x: ndarray, training: bool) -> ndarray:
        self.inputs = x
        self.output = np.maximum(0, self.inputs)
        return self.output

    def backward(self, dvalues: ndarray):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

    def predictions(self, outputs: ndarray) -> ndarray:
        return outputs


class Sigmoid(ActivationFunction):

    def forward(self, inputs: ndarray, training: bool) -> None:
        self.inputs = inputs
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues: ndarray) -> None:
        self.dinputs = self.output * (1 - self.output) * dvalues

    def predictions(self, outputs: ndarray) -> ndarray:
        return (outputs > 0.5) * 1


class Softmax(ActivationFunction):

    def forward(self, x: ndarray | float | int, training: bool) -> None:
        # Get non normalised values
        # Subtract the max to keep all values between 0 and 1, and preventing overflow error
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))

        # Normalise
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backward(self, dvalues: ndarray) -> None:
        # Uninitialised array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (output, dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            output = output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(output) - np.dot(output, output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, dvalues)

    def predictions(self, outputs: ndarray) -> ndarray:
        return np.argmax(outputs, axis=1)


class Linear(ActivationFunction):

    def forward(self, x: ndarray, training: bool) -> None:
        # Only needs to copy values
        self.input = x
        self.output = x

    def backward(self, dvalues: ndarray) -> None:
        # d/dx = 1, 1 * dvalues = dvalues using chain rule
        self.dinputs = dvalues.copy()

class ActivationLoss:
    class ActLoss:

        def forward(self, x: ndarray, y_true: ndarray, training: bool) -> None:
            pass

        def backward(self, dvalues: ndarray, y_true: ndarray) -> None:
            pass

    class ActivationSoftmaxLossCategoricalCrossentropy(ActLoss):

        def __init__(self):
            self.activation = Softmax()
            self.loss = Loss.CategoricalCrossentropy()

        def forward(self, x: ndarray, y_true: ndarray, training: bool):
            self.activation.forward(x)
            self.output = self.activation.output

            return self.loss.calculate(self.output, y_true)

        def backward(self, dvalues: ndarray, y_true: ndarray) -> None:
            samples = len(dvalues)

            if len(y_true.shape) == 2:
                y_true = np.argmax(y_true, axis=1)

            self.dinputs = dvalues.copy()
            self.dinputs[range(samples), y_true] -= 1
            self.dinputs = self.dinputs / samples