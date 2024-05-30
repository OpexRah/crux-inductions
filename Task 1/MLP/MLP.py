import numpy as np
from activation_functions import *

class MLP:
    def __init__(self, layer_list=[], lr = 0.02):
        """
        layer_list: this is a list that contains the number of neurons per layer. length of list = number of layers
        lr: learning rate for the network
        """
        self.hidden_layers = len(layer_list) - 1
        self.weights = self.init_weights(layer_list)
        self.lr = lr

    def init_weights(self, layer_list):
        """
        layer_list: this is a list that contains the number of neurons per layer. length of list = number of layers

        Returns: list of weights
        """
        layers = len(layer_list)
        weights = []

        for i in range(1, layers):
            w = [[np.random.uniform(-1, 1) for k in range(layer_list[i-1] + 1)] for j in range(layer_list[i])]
            weights.append(np.matrix(w))

        return weights

    def forward_prop(self, X, layers):
        """
        X: input data
        layers: number of layers

        Returns: the output values of each layer in the form of a list after applying activation function
        """
        activations = [X]
        layer_input = X

        for i in range(layers):
            activation = Sigmoid(np.dot(layer_input, self.weights[i].T))
            activations.append(activation)
            layer_input = np.append(1, activation)

        return activations
    
    def back_prop(self, Y, activations, layers):
        """
        Y: target output
        activations: output values of each layer in the form of a list after applying activation function
        layers: number of layers

        Returns: None
        """
        final_output = activations[-1]
        error = np.matrix(Y - final_output)

        for i in range(layers, 0 , -1):
            current_activation = activations[i]

            if i > 1:
                previous_activation = np.append(1, activations[i-1])
            else:
                previous_activation = activations[0]

            dloss = np.multiply(error, SigmoidDerivative(current_activation))
            self.weights[i-1] += self.lr * np.multiply(dloss.T, previous_activation)

            w = np.delete(self.weights[i-1], [0], axis=1)
            error = np.dot(dloss, w)
        

    def index_max_output(self, output):
        """
        output: output of the network

        Returns: index of the maximum value in the output (predicted class)
        """
        max = output[0]
        index = 0

        for i in range(len(output)):
            if output[i] > max:
                max = output[i]
                index = i

        return index

    def predict(self, x):
        """
        x: input data

        Returns: predicted output
        """
        layers = len(self.weights)
        x = np.append(1, x)

        activations = self.forward_prop(x, layers)
        final_output = activations[-1].A1
        index = self.index_max_output(final_output)

        y = [0 for i in range(len(final_output))]
        y[index] = 1

        return y


    def accuracy(self, X, Y):
        """
        X: input data
        Y: target output

        Returns: accuracy of the model
        """
        correct = 0

        for i in range(len(X)):
            x, y = X[i], list(Y[i])
            pred = self.predict(x)

            if y == pred:
                correct += 1

        return correct / len(X)


    def fit(self, X,Y, epochs = 100):
        """
        X: input data
        Y: target output
        epochs: number of iterations for training, default is 100

        Returns: None
        """
        layers = len(self.weights)

        for epoch in range(epochs):
            for i in range(len(X)):
                x, y = X[i], Y[i]
                x = np.matrix(np.append(1, x))

                activations = self.forward_prop(x, layers)
                self.back_prop(y, activations, layers)
            
            if epoch % 10 == 0:
                print(f"Epoch: {epoch}/{epochs}")
                print(f"Training Accuracy : {self.accuracy(X, Y)}")
