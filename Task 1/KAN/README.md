# KAN: Kolmogorov–Arnold Networks

Inspired by the Kolmogorov-Arnold representation theorem,  Kolmogorov Arnold Networks (KANs) are promising alternatives to Multi-Layer Perceptrons (MLPs).

While MLPs have fixed activation functions on nodes (“neurons”), KANs have learnable
activation functions on edges (“weights”). KANs have no linear weights at all – every
weight parameter is replaced by a univariate function parametrized as a spline. 

This simple change makes KANs outperform MLPs in terms of accuracy
and interpretability, on small-scale AI + Science tasks. 

### KAN implementation from scratch
The ```KANpython.py``` file contains the implementation for a simple KAN model that can approximate simple functions. ```KAN_python_test.ipynb``` contains the code for testing the implementation on various functions

Here are some visuals of how the model learns some functions:

#### $Sin(x)$
![](https://github.com/OpexRah/crux-inductions/blob/main/Task%201/KAN/Visualisation/sin.gif)

#### $Cos(x)$
![](https://github.com/OpexRah/crux-inductions/blob/main/Task%201/KAN/Visualisation/cos.gif)

#### $e^x$
![](https://github.com/OpexRah/crux-inductions/blob/main/Task%201/KAN/Visualisation/exp.gif)

#### $e^x + e^{-x}$
![](https://github.com/OpexRah/crux-inductions/blob/main/Task%201/KAN/Visualisation/expxplusexpmx.gif)

### KAN implementation with pykan library
The ```KAN_pykan.ipynb``` file contains the implementation and testing with the pykan library. The testing has been done with the iris dataset. The model got an accuracy of about 91% on the test dataset