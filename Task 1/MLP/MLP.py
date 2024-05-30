import numpy as np
import activations

class MLP:
    def __init__(self, nodes=[], lr = 0.02):
        """
        nodes: this is a list that contains the number of neurons per layer. length of list = number of layers
        lr: learning rate for the network
        """
        self.hidden_layers = len(nodes) - 1
        self.weights = self.init_weights(nodes)

    def init_weights(nodes):
        layers = len(nodes)
        weights = []

        for i in range(1, layers):
            w = [[np.random.uniform(-1, 1) for k in range(nodes[i-1] + 1)] for j in range(nodes[i])]
            weights.append(np.matrix(w))