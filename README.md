# CRUX Induction Tasks

### Task 1:
Use the iris dataset for this task.
1.	Read up on Kolomogorov-Arnold Networks and the KA representation theorem. Find out how they differ from how conventional MLPs work.
2.	Implement a KAN in python from scratch and create methods for

    1.	Forward propagation

    2.	Plotting the splines

    3.	Training the KAN from scratch

3.	Implement a KAN using the pykan library
4.	Implement an MLP from scratch and create methods to train it and to generate predictions. (documentation is left)
5.	Compare the two models (MLP from scratch, KAN from pykan). Implement a grid search algo (from scratch) and run the search on the KAN.

### Task 2:
1.	Implement a(n) (bonus: attention-based) LSTM encoder-decoder model using keras/tensorflow/pytorch and fit it to any multivariate time-series dataset of your choice.
2.	Fit a SARIMAX model to the same dataset. Finetune the parameters for a good fit, and compare the two models.

### Task 3:
1.	Using the youtube videos dataset scraped for your previous project, Studily, create a simple sentiment classifier using a (simplified) MAMBA architecture described in [this][1] paper. 
2.	Implement the architecture using pytorch/tensorflow/keras. 
3.	Create a callback to stop training if accuracy does not improve.

## Environment Setup

To set up your environment, follow these steps:


1. Install required packages: Run the following command in terminal to install the necessary packages:

    ```
    pip install -r requirements.txt
    ```
 

2. Verify installation: To verify that everything is set up correctly, run the following command to check the installed packages:

    ```
    pip list
    ```

    This should display a list of installed packages, including the ones specified in the `requirements.txt` file.

### Note
Details of each task is in the respective task folder.

[1]:https://arxiv.org/pdf/2312.00752

