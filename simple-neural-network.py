import numpy as np
import pandas as pd

NN_ARCHITECTURE = [
    {"input_dim": 2, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 50, "activation": "relu"},
    {"input_dim": 50, "output_dim": 25, "activation": "relu"},
    {"input_dim": 25, "output_dim": 1, "activation": "sigmoid"},
]

class neuralNetwork(object):
    """
    input: the input layer,
    layers: number of hidden layers
    """
    def __init__(self, learning_rate, nn_architecture):
        self.nn_architecture = nn_architecture
        self.learning_rate = learning_rate

    def initialize_layers(self, nn_architecture=self.nn_architecture):
        layers = len(nn_architecture)
        parameters = {}

        for index, layer in enumerate(nn_architecture):
            input_size = layer["input_dim"]
            output_size = layer["output_dim"]

            parameters["W" + str(index + 1)] = np.random.randn(output_size, input_size) * 0.1
            parameters["b" + str(index + 1)] = np.random.randn(output_size, 1) * 0.1
        return parameters

    def sigmoid(a):
        return 1/(1 + np.exp(-a))

    def relu(a):
        return np.maximum(0, a)

    def derivative_sigmoid(db, a):
        s = sigmoid(a)
        return db * s * (1 - s)

    def derivative_relu(db, a):
        dA = np.array(db, copy=True)
        dA[a <= 0] = 0
        return dA

    def layer_forward_propagation(activation_previous, W, b, function="sigmoid"):
        Z_matrix = np.dot(W, activation_previous) + b
        if function=="relu":
            return relu(Z_matrix), Z_matrix
        else:
            return sigmoid(Z_matrix), Z_matrix

    def forward_propagation(X, parameters, nn_architecture=self.nn_architecture):
        memory = {}
        activation = X
        for index, layer in enumerate(nn_architecture):
            activation_previous = activation
            function = layer["activation"]
            W = parameters["W" + str(index + 1)]
            b = parameters["b" + str(index + 1)]
            activation, z_matrix = layer_forward_propagation(activation_previous, W, b, function)

            memory["A" + str(index)] = activation_previous
            memory["A" + str(index + 1)] = z_matrix

        return activation, memory

    def layer_backward_propagation(derivative_activation, activation_previous, W, b, z_matrix, function="sigmoid"):
        if function=="relu":
            dz = derivative_relu(derivative_activation, z_matrix)
        else:
            dz = derivative_sigmoid(derivative_activation, z_matrix)

        dW = np.dot(dz, activation_previous.T) / (activation_previous.shape[1])
        db = np.sum(dz, axis=1, keepdims=True) / (activation_previous.shape[1])
        dA_previous = np.dot(W.T, dz)

        return dA_previous, dW, db

    def backward_propagation(y_predicted, y, parameters, memory, nn_architecture=self.nn_architecture):
        values = {}
        dA_previous = -(np.divide(y, y_predicted) - np.divide(1 - y, 1- y_predicted))

        for index, layer in reversed(list(enumerate(nn_architecture))):
            function = layer["activation"]
            dA = dA_previous

            activation_previous = memory["A" + str(index)]
            z_matrix = memory["Z + str(index + 1)]
            W = parameters["W" + str(index + 1)]
            b = parameters["b" + str(index + 1)]

            dA_previous, dW, db = layer_backward_propagation(dA, activation_previous, W, b, z_matrix, function)
            values["dW" + str(index + 1)] = dW
            values["db" + str(index + 1)] = db

        return values

    def update(parameters, values, nn_architecture=self.nn_architecture):
        for index, layer in enumerate(nn_architecture):
            parameters["W" + str(index)] -= self.learning_rate * values["dW" + str(index)]
            parameters["b" + str(index)] -= self.learning_rate * values["dW" + str(index)]
        return parameters

    def train(X, y, epochs, nn_architecture=self.nn_architecture):
        parameters = initialize_layers(nn_architecture, 2)

        for i in range(epochs):
            y_predicted, memory = forward_propagation(X, parameters, nn_architecture)
            values = backward_propagation(y_predicted, y, parameters, memory, nn_architecture)
            parameters = update(parameters, values, nn_architecture)

        return parameters

# https://towardsdatascience.com/lets-code-a-neural-network-in-plain-numpy-ae7e74410795
