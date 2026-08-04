"""
Neural network: Multi-Layer Perceptron.
"""


""" Third-party packages """
import numpy as np

""" mlpack resources """
from mlpack.model import Model, VerbosityLevel


class MLP(Model):
    def __init__(self, 
        input_size, 
        hidden_size, 
        output_size, 
        learning_rate=0.01, 
        patience=100, 
        batch_size=32,
        *args, **kwargs
    ):
        
        super().__init__(*args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.eta = learning_rate
        self.patience = patience

        self.batch_size = batch_size

        # Weights for the input - hidden layer step
        self.weights_hidden = np.random.randn(input_size, hidden_size) * 0.01
        self.bias_hidden = np.zeros((1, hidden_size))

        # Weights for the hidden layer - output step
        self.weights_output = np.random.randn(hidden_size, output_size) * 0.01
        self.bias_output = np.zeros((1, output_size))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        z = np.exp(x - np.max(x, axis=1, keepdims=True))
        return z / np.sum(z, axis=1, keepdims=True)

    def forward(self, X):
        # Compute hidden layer step
        self.hidden_layer_input = np.dot(X, self.weights_hidden) + self.bias_hidden
        self.hidden_layer_output = self.sigmoid(self.hidden_layer_input)

        # Compute output layer step
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_output) + self.bias_output
        self.output_layer_output = self.softmax(self.output_layer_input)

        # Return output
        return self.output_layer_output

    def backward(self, X, y, output):
        num_samples = X.shape[0]

        # Gradient of the loss function with respect to the output layer
        d_output = output - y

        # Gradient of the output layer weights and biases
        d_weights_output = np.dot(self.hidden_layer_output.T, d_output) / num_samples
        d_bias_output = np.sum(d_output, axis=0, keepdims=True) / num_samples

        # Gradient of the hidden layer
        d_hidden = np.dot(d_output, self.weights_output.T) * self.sigmoid_derivative(self.hidden_layer_output)

        # Gradient of the hidden layer weights and biases
        d_weights_hidden = np.dot(X.T, d_hidden) / num_samples
        d_bias_hidden = np.sum(d_hidden, axis=0, keepdims=True) / num_samples

        return d_weights_hidden, d_bias_hidden, d_weights_output, d_bias_output

    def update_parameters(self, d_weights_hidden, d_bias_hidden, d_weights_output, d_bias_output):
        self.weights_hidden -= self.eta * d_weights_hidden
        self.bias_hidden -= self.eta * d_bias_hidden
        self.weights_output -= self.eta * d_weights_output
        self.bias_output -= self.eta * d_bias_output

    def fit(self, X_train, y_train):
        num_classes = self._get_num_classes(y_train)
        y_train = np.eye(num_classes)[y_train.reshape(-1)]
        num_samples = X_train.shape[0]
        num_batches = num_samples // self.batch_size

        for epoch in range(self.patience):
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for i in range(0, num_samples, num_batches):
                X_batch = X_shuffled[i:i+num_batches]#start:end]
                y_batch = y_shuffled[i:i+num_batches]#start:end]

                output = self.forward(X_batch)
                d_wh, d_bh, d_wo, d_bo = self.backward(X_batch, y_batch, output)
                self.update_parameters(d_wh, d_bh, d_wo, d_bo)

            if self.verbosity is not VerbosityLevel.SILENCED:
                print(f"Epoch: {epoch+1}/{self.patience}", end='\r')

    def predict(self, X):
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        return predictions

    def __str__(self):
        return f"MLP(input_size={self.input_size}, hidden_size={self.hidden_size}, output_size={self.output_size}, learning_rate={self.eta}, patience={self.patience}, batch_size={self.batch_size}, weights_hidden={self.weights_hidden}, bias_hidden={self.bias_hidden}, weights_output={self.weights_output}, bias_output={self.bias_output})"

    def __repr__(self):
        return self.__str__()

#MLP
