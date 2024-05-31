import numpy as np
import pandas as pd
import torch
from kan import *
import itertools


def gridsearch(data, params, classifier):

    """
    Function to perform grid search on the given parameters for the given classifier
    data: pandas dataframe, the dataset to be used for training and testing
    params: dictionary, the parameters to be used for grid search
    classifier: the classifier to be used for training and testing

    Returns: dictionary of results of the grid search
    """

    result_params = {'params': list(itertools.product(*[params[k] for k in params])), 'mean_train_score': [], 'mean_test_score': []}

    for param in result_params['params']:
        print(f"Training model with width: {param[0]}, grid: {param[1]}, k: {param[2]}")
        dummy_classifier = classifier(width=param[0], grid=param[1], k=param[2], seed=0)
        classifier_train_score = []
        classifier_test_score = []

        def train_acc():
            return torch.mean((torch.argmax(dummy_classifier(dataset['train_input']), dim=1) == dataset['train_label']).float())
        
        def test_acc():
            return torch.mean((torch.argmax(dummy_classifier(dataset['test_input']), dim=1) == dataset['test_label']).float())

        for fold in range(5):
            test_fold_0 = data[data["variety"] == 0.0][fold*10:(fold+1)*10]
            test_fold_1 = data[data["variety"] == 1.0][fold*10:(fold+1)*10]
            test_fold_2 = data[data["variety"] == 2.0][fold*10:(fold+1)*10]
            test_fold = pd.concat([test_fold_0, test_fold_1, test_fold_2])

            train_fold = data.drop(test_fold.index)

            dataset = {}
            
            dataset["train_input"] = torch.from_numpy(np.array(train_fold.iloc[:, :-1].values)).float()
            dataset["train_label"] = torch.from_numpy(np.array(train_fold.iloc[:, -1].values)).type(torch.LongTensor)
            dataset["test_input"] = torch.from_numpy(np.array(test_fold.iloc[:, :-1].values)).float()
            dataset["test_label"] = torch.from_numpy(np.array(test_fold.iloc[:, -1].values)).type(torch.LongTensor)

            results = dummy_classifier.train(dataset, opt="LBFGS", steps=20, metrics=(train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss());

            classifier_train_score.append(results['train_acc'][-1])
            classifier_test_score.append(results['test_acc'][-1])

        result_params['mean_train_score'].append(np.mean(classifier_train_score))
        result_params['mean_test_score'].append(np.mean(classifier_test_score))

    return result_params

# Load the iris dataset and prepare it for KAN model

df = pd.read_csv("Task 1/Dataset/iris.csv")
df["variety"].replace(["Setosa", "Versicolor", "Virginica"], [0., 1., 2.], inplace=True) #replace the categorical labels with numbers
df = df.sample(frac=1, random_state=0).reset_index(drop=True) #shuffle the dataset


params = {'width': [[4,3], [4,10,3]], 'grid': [3, 10], 'k': [3, 4]}

result = gridsearch(df, params, classifier=KAN)

best_params = result['params'][np.argmax(result['mean_test_score'])]
best_accuracy = np.max(result['mean_test_score'])

print(result)

print(best_params, best_accuracy)
