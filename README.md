# DataPrivacy

- Description
- Used Technologies

## Description
Those scripts shows how to strengthen privacy. There are used two methods: perturbation method (adding noise to original data) and encrypting (perturbated data is encrypthed). Using perturbation method, the privacy of original data and the privacy of the model based on original data are protected. Script using perturbation method (data_privacy.py) also shows the differences between created models: one using original data and the second one, using synthetic data. Used dataset comes from sklearn.datasets and is named load_breast_cancer.

## Used Technologies
1. Programming language: Python
2. Used libraries: numpy, sklearn.datasets, sklearn.model_selection, sklearn.neural_network, matplotlib.pyplot, pandas, cryptography.fernet
