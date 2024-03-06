import numpy as np
from numpy import ndarray
import Layers

class LossFunction:

    def forward(self, y_pred: ndarray, y_true: ndarray) -> None:
        """
        Calculates the loss of a neural network given the predicted values and true values
        :param y_pred: Predicted values
        :param y_true: True values
        :return: None. Outputs are stored in the self.outputs attribute
        """
        pass

    def backward(self, dvalues: ndarray, y_true: ndarray) -> None:
        """
        Backwards pass using the derivative of the loss function
        :param dvalues: Predicted values
        :param y_true: True values
        :return: None. Outputs are stored in self.dinputs
        """

    def calculate(self, output: ndarray, y: ndarray, *, include_regularization: bool = False):
        self.forward(output, y)
        sample_losses = self.output
        data_loss = np.mean(sample_losses)

        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)

        if not include_regularization:
            return data_loss, 0
        return data_loss, self.regularization_loss()

    def regularization_loss(self) -> float:
        # 0 by default
        regularization_loss = 0

        for layer in self.trainable_layers:

            if isinstance(layer, Layers.LayerDense):

                # Only calculate values if factor > 0
                # L1 regularization - weights
                if layer.weight_regularizer_l1 > 0:
                    regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

                # L2 regularization - weights
                if layer.weight_regularizer_l2 > 0:
                    regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

                # L1 - biases
                if layer.bias_regularizer_l1 > 0:
                    regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

                # L2 - biases
                if layer.bias_regularizer_l2 > 0:
                    regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

    def remember_trainable_layers(self, trainable_layers: list) -> None:
        # Sets layers that can be trained in the model
        self.trainable_layers = trainable_layers

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate_accumulated(self, *, include_regularization: bool = False):
        # Mean loss
        data_loss = self.accumulated_sum / self.accumulated_count

        if not include_regularization:
            return data_loss
        return data_loss, self.regularization_loss()


class CategoricalCrossentropy(LossFunction):

    def forward(self, y_pred: ndarray, y_true: ndarray) -> None:
        samples = len(y_pred)
        # Done to prevent any 0 values as log(0) is taken as infinity
        y_pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)

        if len(y_true.shape) == 1:
            correct_confidence = y_pred_clipped[range(samples), y_true]

        elif len(y_true.shape) == 2:
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        else:
            y_true = y_true.reshape(y_true.shape[0], -1)
            correct_confidence = np.sum(y_pred_clipped * y_true, axis=1)

        self.output = -np.log(correct_confidence)

    def backward(self, dvalues: ndarray, y_true: ndarray) -> None:
        # derivative of -sum(log(x)) = -sum(y_pred/y_true)
        samples = len(dvalues)
        labels = len(dvalues[0])

        # convert into one-hot vector if sparse
        if len(y_true.shape) == 1:
            # np.eye takes in n and returns n x n matrix with leading diagonal as 1
            y_true = np.eye(labels)[y_true]

        self.dinputs = -y_true / dvalues
        self.dinputs = self.dinputs / samples


class BinaryCrossentropy(LossFunction):

    def forward(self, y_pred: ndarray, y_true: ndarray) -> None:
        # clip data to prevent log(0)
        # clip both sides to not affect mean
        y_pred_clipped = np.clip(y_pred, 1e-12, 1 - 1e-12)

        # calculate loss per sample
        sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        self.output = sample_losses

    def backward(self, dvalues: ndarray, y_true: ndarray) -> None:
        samples = len(dvalues)
        # No. of outputs per samples, using first sample to calculate
        outputs = len(dvalues[0])

        # clip to prevent log(0)
        clipped_dvalues = np.clip(dvalues, 1e-12, 1 - 1e-12)

        # calculate gradient
        self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
        # normalise
        self.dinputs = self.dinputs / samples


class MeanSquaredError(LossFunction):

    def forward(self, y_pred: ndarray, y_true: ndarray) -> None:
        # Squaring the difference between true and predicted values across each input
        self.output = np.mean((y_true - y_pred) ** 2, axis=-1)

    def backward(self, dvalues: ndarray, y_true: ndarray) -> None:
        # No. of samples
        samples = len(dvalues)
        # No. of outputs per sample
        outputs = len(dvalues[0])

        # Gradient calculation and normalization
        self.dinputs = -2 * (y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples


class MeanAbsoluteError(LossFunction):

    def forward(self, y_pred: ndarray, y_true: ndarray) -> None:
        # Mean of the absolute difference between true values and predicted values per input
        self.output = np.mean(np.abs(y_true - y_pred), axis=-1)

    def backward(self, dvalues: ndarray, y_true: ndarray) -> None:
        # No. of samples
        samples = len(dvalues)
        # No. of outputs per sample
        outputs = len(dvalues[0])

        # Calculate and normalize gradient
        self.dinputs = np.sign(y_true - dvalues) / outputs
        self.dinputs = self.dinputs / samples
