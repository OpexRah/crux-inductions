# MLP: Multi-Layer Perceptron
The MLP (Multi-Layer Perceptron) is a type of artificial neural network that consists of multiple layers of interconnected neurons. It is a feedforward neural network, meaning that the information flows in one direction, from the input layer to the output layer.

The implementation of MLP is provided in the `MLP.py` file. The code defines a class called `MLP` that represents the MLP model. It has methods for initializing the model, performing forward propagation, backpropagation, prediction, and calculating accuracy.

The `__init__` method initializes the MLP model with the specified number of neurons per layer and learning rate. The `init_weights` method initializes the weights of the MLP model randomly. The `forward_prop` method performs forward propagation by applying the sigmoid activation function to the weighted sum of inputs at each layer. The `back_prop` method performs backpropagation to update the weights based on the error between the predicted output and the target output. The `index_max_output` method returns the index of the maximum value in the output, which represents the predicted class. The `predict` method predicts the output for a given input. The `accuracy` method calculates the accuracy of the model on a given dataset. The `fit` method trains the model by iterating over the dataset for a specified number of epochs and updating the weights using backpropagation.

To use the MLP model, you can create an instance of the `MLP` class and call the `fit` method to train the model on your dataset. After training, you can use the `predict` method to make predictions on new data.

Please note that the code provided is a simplified implementation of MLP and may not include all the advanced features and optimizations commonly used in practice.

## Testing with iris dataset
The code for testing the MLP model with the iris dataset can be found in ```MLP_test.ipynb```. It got an accuracy of about 99% on the testing dataset.