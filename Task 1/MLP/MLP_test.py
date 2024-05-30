import numpy as np
import pandas as pd
from MLP import MLP

df = pd.read_csv("Task 1/Dataset/iris.csv")
df["variety"].replace(["Setosa", "Versicolor", "Virginica"], [0, 1, 2], inplace=True)
df = df.sample(frac=1).reset_index(drop=True)

X = np.array(df.iloc[:, :-1].values)
Y = np.array(df.iloc[:, -1].values)
Y = pd.get_dummies(Y).values

train_split = 0.85

X_train = X[:int(len(X)*train_split)]
Y_train = Y[:int(len(Y)*train_split)]
X_test = X[int(len(X)*train_split):]
Y_test = Y[int(len(Y)*train_split):]


model = MLP([4,5,10,3], 0.15)
model.fit(X_train, Y_train, epochs=100)

print(f"Testing accuracy: {model.accuracy(X_test, Y_test)}")