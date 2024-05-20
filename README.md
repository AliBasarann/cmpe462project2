# cmpe462hw1
This repository includes the source code for CMPE 462 HW1. </br>
## Install requirements
`pip install -r requirements.txt` </br>
## Install the dataset
- We used the csv version of the MNIST dataset. You can load the csv version of the MNIST dataset from the following [link](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- The wdbc.data (Breast Cancer Wisconsin Diagnostic dataset) should be put in the data folder.
## decision_tree.py
This code was implemented for training a decision tree for MNIST and WDBC dataset in the data folder. In order to run the code, run `python decision_tree.py`. </br>
## svm.py
This code was implemented for training an SVM for MNIST in the data folder. In order to run the code without feature extraction, run `python svm.py`. If you want to run the code with feature extraction, run `python svm.py true`</br>
## kmeans.py
This code was implemented for kmeans clustering for MNIST dataset in the data folder. In order to run the code, run `python kmeans.py`.

