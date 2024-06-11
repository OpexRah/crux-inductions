# MAMBA
Mamba is a new state space model architecture showing promising performance on information-dense data such as language modeling. It is much faster (subquadratic) than the transformer which is an O($n^2$) model.

The ```MAMBA.py``` file contains the implementation of a simple MAMBA architecture as described in [this][1] paper. The model is then tested on the YouTube dataset which contains titles of YouTube videos and its corresponding Class (Educational / Non-Educational). The task is to classify the title as Educational or Non-Educational

You can find the results in the ```MAMBA_train.ipynb``` file.

I have also added an early stopping function to stop the training in case accuracy does not improve. It stops the training if no improvement in accuracy is seen after a couple of epochs.

[1]:https://arxiv.org/pdf/2312.00752