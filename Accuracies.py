import numpy as np
from numpy import ndarray


class Accuracy:

    def calculate(self, predictions: ndarray, y: ndarray) -> float:
        comparisons = self.compare(predictions, y)
        accuracy = np.mean(comparisons)

        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return accuracy

    def compare(self, predictions: ndarray, y: ndarray) -> ndarray:
        """
        Returns comparison between actual values and predictions
        :param predictions: Predicted values
        :param y: Actual values
        :return: NumPy array of predictions
        """

    def new_pass(self):
        # Reset variables for accumulated accuracy
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate_accumulated(self):
        accuracy = self.accumulated_sum / self.accumulated_count
        return accuracy


class RegressionAccuracy(Accuracy):

    def __init__(self):
        self.precision = None

    def init(self, y: ndarray, reinit: bool = False) -> None:
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    def compare(self, predictions: ndarray, y: ndarray) -> ndarray:
        return np.absolute(predictions - y) < self.precision


class CategoricalAccuracy(Accuracy):

    def __init__(self, *, binary: bool = False):
        # Determine if binary categorical or not
        self.binary = binary

    def init(self, y: ndarray, reinit: bool = False) -> None:
        # Not necessary for this class but will be called anyway
        pass

    def compare(self, predictions: ndarray, y: ndarray) -> ndarray:
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)
        result = np.array(predictions) == np.array(y)
        return result

