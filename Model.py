import numpy as np
from numpy import ndarray
import Optimizers
import Loss
import Accuracies
import ActivationFunctions
import Layers
import pickle
import copy


class Model:

    def __init__(self):
        # List of layers
        self.layers = []
        self.trainable_layers = []
        # Whether or not there is a combined softmax classifier object
        self.softmax_classifier_output = None

    def set(self, *, loss: Loss.LossFunction = None, optimizer: Optimizers.Optimizer = None,
            accuracy: Accuracies.Accuracy = None):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def add(self, layer) -> None:
        self.layers.append(layer)

    def train(self, X: ndarray, y: ndarray, *, epochs: int = 1, batch_size: int = None,
              print_every: int = 1, validation_data=None):
        self.accuracy.init(y)
        train_steps = 1

        if validation_data is not None:
            validation_steps = 1
            X_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size
            # Add 1 to include all the data
            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val)
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        for epoch in range(1, epochs + 1):
            print(f'Epoch: {epoch}')

            # Reset accumulated loss and accuracy values
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(train_steps):
                # If batch size not set use full dataset
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step * batch_size:(step + 1) * batch_size]
                    batch_y = y[step * batch_size:(step + 1) * batch_size]

                # Forward pass
                output = self.forward(batch_X, training=True)

                # Calculate loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                predictions = self.output_layer.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                self.backward(output, batch_y)

                # Optimization
                self.optimizer.decay_pre_update()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update()

                # Print summary
                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.3f} (' +
                          f'data_loss: {data_loss:.3f}, ' +
                          f'reg_loss: {regularization_loss:.3f}), ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training, ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.3f} (' +
                  f'data_loss: {epoch_data_loss:.3f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.3f}), ' +
                  f'lr: {self.optimizer.current_learning_rate}')

        if validation_data is not None:
            self.evaluate(*validation_data, batch_size=batch_size)

    def evaluate(self, X_val: ndarray, y_val: ndarray, *, batch_size=None):
        # Default value
        validation_steps = 1

        # calculate number of steps
        if batch_size is not None:
            validation_steps = len(X_val) // batch_size
            # Add 1 if not enough steps due to integer division
            if validation_steps * batch_size < len(X_val):
                validation_steps += 1

        # Reset accumulated loss and accuracy values
        self.loss.new_pass()
        self.accuracy.new_pass()

        for step in range(validation_steps):
            # If batch size not set use full dataset
            if batch_size is None:
                batch_X = X_val
                batch_y = y_val
            else:
                batch_X = X_val[step * batch_size:(step + 1) * batch_size]
                batch_y = y_val[step * batch_size:(step + 1) * batch_size]

            # Forward pass
            output = self.forward(batch_X, training=True)

            # Calculate loss, get predictions and calculate accuracy
            self.loss.calculate(output, batch_y)
            predictions = self.output_layer.predictions(output)
            self.accuracy.calculate(predictions, batch_y)

        # Final loss and accuracy calculation
        validation_loss = self.loss.calculate_accumulated()
        validation_accuracy = self.accuracy.calculate_accumulated()

        print(f'validation, ' +
              f'acc: {validation_accuracy:.3f}, ' +
              f'loss: {validation_loss:.3f}')

        return validation_loss

    def finalize(self) -> None:
        self.input_layer = Layers.Input()
        # count of hidden layers
        layer_count = len(self.layers)

        for i in range(layer_count):
            # If it's the first layer, the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]
            # All layers except for first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]
            # Last layer, next object is loss
            # Also save reference to last object which will be model's output
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer = self.layers[i]

            # If a layer contains weights, it can be tweaked as it isn't an activation layer
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        # If a model is loaded it won't have a loss object so only needs to be done if used for training
        if self.loss is not None:
            self.loss.remember_trainable_layers(self.trainable_layers)

        # If output is softmax and loss is categorical crossentropy create combined object for SPEED
        if (isinstance(self.layers[-1], ActivationFunctions.Softmax) and
                isinstance(self.loss, Loss.CategoricalCrossentropy)):
            self.softmax_classifier_output = (
                ActivationFunctions.ActivationLoss.ActivationSoftmaxLossCategoricalCrossentropy())

    def forward(self, X: ndarray, *, training: bool) -> ndarray:
        # Sets output property that first layer can access in prev attribute
        self.input_layer.forward(X, training)

        # Chain call forward method of each object and pass previous output as parameter
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # Layer is now last layer
        return layer.output

    def backward(self, output: ndarray, y: ndarray) -> None:
        # If combined loss and activation object
        if self.softmax_classifier_output is not None:
            # Call backward to send dinputs
            self.softmax_classifier_output.backward(output, y)

            # Set dinputs of last layer as it won't be called
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            # Call backward going through objects but last
            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # First call on loss to set dinputs property
        self.loss.backward(output, y)

        # Call backward going through layers in reverse
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)

    def get_parameters(self) -> list[tuple[ndarray]]:
        # Retrieves parameters of trainable layers
        parameters = []
        for layer in self.trainable_layers:
            parameters.append(layer.get_paramters())
        return parameters

    def set_parameters(self, parameters: list[tuple[ndarray]]):
        # Iterate over parameters and layers and update layers with parameters
        for parameter_set, layer in zip(parameters, self.trainable_layers):
            layer.set_parameters(*parameter_set)

    def save_parameters(self, path: str):
        # Open a file with binary-write and save parameters
        with open(path, 'wb') as f:
            pickle.dump(self.get_parameters(), f)

    def load_parameters(self, path: str):
        # Open file with binary-read and load weights and update layers
        with open(path, 'rb') as f:
            self.set_parameters(pickle.load(f))

    def save(self, path: str):
        # Saves the entire model
        model = copy.deepcopy(self)

        # Reset accumulated loss and accuracy values
        model.loss.new_pass()
        model.accuracy.new_pass()

        # Remove data from input layer and gradients from loss object
        model.input_layer.__dict__.pop('output', None)
        model.loss.__dict__.pop('dinputs', None)

        # Remove inputs, output, and dinputs from layers
        for layer in model.layers:
            for property in ['inputs', 'output', 'dinputs', 'dweights', 'dbiases']:
                layer.__dict__.pop(property, None)

        # Save model with binary-write
        with  open(path, 'wb') as f:
            pickle.dump(model, f)

    @staticmethod
    def load(path: str):
        # Load using binary-read mode
        with open(path, 'rb') as f:
            model = pickle.load(f)

        return model

    def predict(self, X: ndarray, *, batch_size=None) -> ndarray:
        # Default value if batch size not set
        prediction_steps = 1

        # Calculate number of steps
        if batch_size is not None:
            prediction_steps = len(X) // batch_size
            # Int division rounds down so add 1 to compensate
            if prediction_steps * batch_size < len(X):
                prediction_steps += 1

        output = []

        for step in range(prediction_steps):
            # If batch size not set use full dataset
            if batch_size is None:
                batch_X = X
            # Otherwise slice
            else:
                batch_X = X[step * batch_size:(step + 1) * batch_size]

            # Forward pass and append to outputs
            batch_output = self.forward(batch_X, training=False)
            output.append(batch_output)

        # Vstack stacks vertically to ensure a 1d array of outputs
        return np.vstack(output)

