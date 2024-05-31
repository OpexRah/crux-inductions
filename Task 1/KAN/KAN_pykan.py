from kan import *
import pandas as pd
import numpy as np
import torch

# Load the iris dataset and prepare it for KAN model

df = pd.read_csv("Task 1/Dataset/iris.csv")
df["variety"].replace(["Setosa", "Versicolor", "Virginica"], [0., 1., 2.], inplace=True) #replace the categorical labels with numbers
df = df.sample(frac=1, random_state=0).reset_index(drop=True) #shuffle the dataset

X = np.array(df.iloc[:, :-1].values)
Y = np.array(df.iloc[:, -1].values)
X = torch.from_numpy(X).float() # KAN needs its data in torch.tensor dtype
Y = torch.from_numpy(Y).type(torch.LongTensor) # CrossEntropyLoss needs the labels to be in integer dtype

train_split = 0.85

X_train = X[:int(len(X)*train_split)]
Y_train = Y[:int(len(Y)*train_split)]
X_test = X[int(len(X)*train_split):]
Y_test = Y[int(len(Y)*train_split):]

dataset = {}
dataset["train_input"] = X_train
dataset["test_input"] = X_test
dataset["train_label"] = Y_train
dataset["test_label"] = Y_test

"""
KAN model with 3 hidden layers of width 4, 5, and 3 respectively, grid size of 3, and k=3
"""
model = KAN(width=[4,5,3], grid=3, k=3, seed=0)

"""
define training and testing accuracy functions which will be used as metrics for KAN.train() method
"""

def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())

def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())

# train the model
results = model.train(dataset, opt="LBFGS", steps=20, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss());

#print results
print(f"Train Accuracy: {results['train_acc'][-1]}")
print(f"Test Accuracy: {results['test_acc'][-1]}")

"""
Pykan also allows us to see the symbolic formula of the model. In this case we have 3 formulas each corresponding to a class.
Each formula has 4 variables which correspond to our 4 features in the iris dataset. Plugging in the values of these variables will 
give us the probability of the input belonging to that class.

Note that the formulas generated are not the same as the ones used to generate the data. (ideal function) 
They are just and approximation of the function that the model has learned and is true upto some error threshold given by 
the Kolmogorov-Arnold representation theorem.
"""

symbols = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs'] #symbols that can be used in the formula
model.auto_symbolic(lib=symbols) #generate the symbolic formula


formula1, formula2, formula3 = model.symbolic_formula()[0]
print(f"Formula 1 :{formula1}\n\nFormula 2:{formula2}\n\nFormula 3:{formula3}")