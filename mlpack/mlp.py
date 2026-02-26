import numpy as np

#class MultiLayerPerceptron:
#    def __init__(self, input_size, hidden_size, output_size):
#        self.weights_input_hidden = np.random.randn(input_size, hidden_size)
#        self.weights_hidden_output = np.random.randn(hidden_size, output_size)
#        self.bias_hidden = np.zeros((1, hidden_size))
#        self.bias_output = np.zeros((1, output_size))
#
#    def sigmoid(self, z):
#        return 1 / (1 + np.exp(-z))
#
#    def softmax(self, x):
#        z = np.exp(x - np.max(x))
#        return z / z.sum(axis=1, keepdims=True)
#
#    def forward(self, X: np.ndarray):
#        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
#        self.hidden_output = self.sigmoid(self.hidden_input)
#
#        self.final_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
#        self.final_output = self.softmax(self.final_input)
#
#        return self.final_output
#
#    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray, learning_rate):
#        output_error = output - y
#        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.hidden_output * (1 - self.hidden_output)
#        
#        self.weights_hidden_output -= learning_rate * np.dot(self.hidden_output.T, output_error)
#        self.bias_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)
#        self.weights_input_hidden -= learning_rate * np.dot(X.T, hidden_error)
#        self.bias_hidden -= learning_rate * np.sum(hidden_error, axis=0, keepdims=True)
#
#    def fit(self, X_train: np.ndarray, y_train: np.ndarray, max_epochs, learning_rate):
#        for _ in range(max_epochs):
#            output = self.forward(X_train)
#            self.backward(X_train, y_train, output, learning_rate)
#
#    def predict(self, X: np.ndarray):
#        output = self.forward(X)
#        return np.argmax(output, axis=1)
#
#class Neural_Network:
#    def __init__(self, n_in, n_hidden, n_out, max_epochs=100, learning_rate=1.2):
#        # Network dimensions
#        self.n_x = n_in
#        self.n_h = n_hidden
#        self.n_y = n_out
#
#        self.max_epochs = max_epochs
#        self.eta = learning_rate
#        
#        # Parameters initialization
#        self.W1 = np.random.randn(self.n_h, self.n_x) * 0.01
#        self.b1 = np.zeros((self.n_h, 1))
#        self.W2 = np.random.randn(self.n_y, self.n_h) * 0.01
#        self.b2 = np.zeros((self.n_y, 1))
#    
#    def sigmoid(self, z):
#        """
#        Sigmoid function applied to `z`.
#        """
#        #if z >= 0:
#        return 1 / (1 + np.exp(-z))
#        #else:
#        #    return np.exp(z) / (1 + np.exp(z))
#
#
#    def forward(self, X):
#        """ Forward computation """
#        self.Z1 = self.W1.dot(X.T) + self.b1
#        self.A1 = np.tanh(self.Z1)
#        self.Z2 = self.W2.dot(self.A1) + self.b2
#        self.A2 = self.sigmoid(self.Z2)
#    
#    def back_prop(self,  X, Y):
#        """ Back-progagate gradient of the loss """
#        m = X.shape[0]
#        self.dZ2 = self.A2 - Y
#        self.dW2 = (1 / m) * np.dot(self.dZ2, self.A1.T)
#        self.db2 = (1 / m) * np.sum(self.dZ2, axis=1, keepdims=True)
#        self.dZ1 = np.multiply(np.dot(self.W2.T, self.dZ2), 1 - np.power(self.A1, 2))
#        self.dW1 = (1 / m) * np.dot(self.dZ1, X)
#        self.db1 = (1 / m) * np.sum(self.dZ1, axis=1, keepdims=True)
#
#    def fit(self, X, Y):
#        """ Complete process of learning, alternates forward pass,
#            backward pass and parameters update """
#        m = X.shape[0]
#        for _ in range(self.max_epochs):
#            self.forward(X)
#            #loss = -np.sum(np.multiply(np.log(self.A2), Y) + np.multiply(np.log(1-self.A2),  (1 - Y))) / m
#            self.back_prop(X, Y)
#
#            self.W1 -= self.eta * self.dW1
#            self.b1 -= self.eta * self.db1
#            self.W2 -= self.eta * self.dW2
#            self.b2 -= self.eta * self.db2
#
#            #if e % 1000 == 0:
#            #    print("Loss ",  e, " = ", loss)
#
#    def predict(self, X):
#        """ Compute predictions with just a forward pass """
#        self.forward(X)
#        return np.round(self.A2).astype(int)
#from sklearn.datasets import load_digits
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import StandardScaler
#from sklearn.metrics import accuracy_score
 
class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, patience=100, batch_size=32):
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
        num_classes = len(np.unique(y_train))
        y_train = np.eye(num_classes)[y_train.reshape(-1)]
        num_samples = X_train.shape[0]
        num_batches = num_samples // self.batch_size

        for _ in range(self.patience):
            permutation = np.random.permutation(num_samples)
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]

            for i in range(0, num_samples, num_batches):
                #start = i * self.batch_size
                #end = (i + 1) * self.batch_size
                X_batch = X_shuffled[i:i+num_batches]#start:end]
                y_batch = y_shuffled[i:i+num_batches]#start:end]

                output = self.forward(X_batch)
                d_wh, d_bh, d_wo, d_bo = self.backward(X_batch, y_batch, output)
                self.update_parameters(d_wh, d_bh, d_wo, d_bo)

            #if (epoch + 1) % 10 == 0:
            #    output_train = self.forward(X_train)
            #    predictions_train = np.argmax(output_train, axis=1)
            #    accuracy_train = accuracy_score(np.argmax(y_train, axis=1), predictions_train)
            #    print(f"Epoch {epoch+1}/{self.patience}, Training Accuracy: {accuracy_train:.4f}")

    def predict(self, X):
        output = self.forward(X)
        predictions = np.argmax(output, axis=1)
        return predictions
