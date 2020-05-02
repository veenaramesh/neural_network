import numpy as np
import h5py

def initialize_parameters(layer_dims):
    """
    layer_dims -  array containing dimensions of each layer

    RETURNS
    parameters - dict containing parameters, formatted like "WL" and "bL",
    where L stands for the layer number
        WL - weight matrix of shape (layer_dims[L], layer_dims[L-1])
        bL - bias vector of shape (layer_dims[L], 1)
    """

    parameters = {}
    number_of_layers = len(layer_dims)

    for l in range(1, number_of_layers):
        parameters["W"+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))

    return parameters

def linear_forward(A_prev, W, b, activation):
    if activation == "sigmoid":
        Z = np.dot(W, A) + b
        linear_cache = (A, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z = np.dot(W, A) + b
        linear_cache = (A, W, b)
        A, activation_cache = relu(Z)

    return A, (linear_cache, activation_cache)

def forward_propagation(X, parameters):
    """
    X - data as an array (number of features, number of examples) or (n, m)
    parameters - dict containing parameters

    RETURNS
    AL - last activation val
    caches - list of activation values
    """
    caches = []
    A = X
    number_of_layers = len(parameters) // 2
    for l in range(1, number_of_layers):
        A_prev = A
        A, cache = linear_forward(A_prev, parameters["W"+str(l)],
                                          parameters["b"+str(l)],
                                          activation="relu")
        caches.append(cache)

    AL, cache = linear_forward(A, parameters["WL"], parameters["bL"], activation="sigmoid")
    caches.append(cache)

    return AL, caches

def compute_cost(AL, Y):
    """
    AL - probability vector
    y - true label

    RETURNS
    cost - cost-entropy cost
    """
    m = Y.shape[1]
    cost = np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1 - Y), np.log(1 - AL)))/-m

    cost = np.squeeze(cost)
    return cost

def linear_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation = "relu":
        dZ = relu_backward(dA, activation_cache)
        dW = np.dot(dZ, A_prev.T)/m
        db = np.sum(dZ, axis=1, keepdims=True)/m
        dA_prev = np.dot(W.T, dZ)
    elif activation = "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dW = np.dot(dZ, A_prev.T)/m
        db = np.sum(dZ, axis=1, keepdims=True)/m
        dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db

def backward_propagation(AL, Y, caches):
    grads = {}
    number_of_layers = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(dAL, current_cache, activation = "sigmoid")
    for l in reversed(range(number_of_layers-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_backward(grads["dA"+str(l+1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_params(parameters, grads, learning_rate):
    number_of_layers = len(parameters) // 2

    for l in range(number_of_layers):
        parameters["W" + str(l+1)] = parameters["W"+str(l+1)] - learning_rate*grads["dW"+str(l+1)]
        parameters["b" + str(l+1)] = parameters["b"+str(l+1)] - learning_rate*grads["db"+str(l+1)]
    return parameters
