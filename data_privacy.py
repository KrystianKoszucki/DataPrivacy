#This script is used for protecting the original database and neural network model made of it. The script shows the differences between the model using original data
#and models, which were build of synthetic database made of adding normal noise, cauchy noise, poisson noise to the original database. That way of protecting data is
#called "perturbative method". In general, this method uses only a noise comming from the normal distribution, but there were used the other ones to check, if other 
#distibutions could be usefull.
#To show the diffrences:
# 1. the script computes how many times each model makes a mistake,
# 2. the script draws heat map of input features (if importances of input features changes, the models will be different
#Whole script is object-oriented.


#importing libraries and functions
#numpy stands for doing calculations on matrices and for creating normal, poisson and cauchy noisses.
#load_breast_cancer is the database used as original
#function train_test_split stands for seperating the data into 2 parts- training data and testing data. It is necessary for building machine learning model.
#MLPClassifier is a class of neural network, which is our machine learning model.
#matplotlib.pyplot is used for ploting heat maps
#pandas was used to change representation of databases by DataFrame class.

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd



#assigning database load_breast_cancer() to variable data_cancer
data_cancer=load_breast_cancer()



#body of script
#Model is a class, which is using to build objects 
class Model:
    def __init__(self, name, database, data) :
        self.name = name
        self.database = database
        self.data= data

        
# print_data method is used for changing representation of data and showing it to the user
    def print_data(self):
        presented_data=np.c_[self.data.data, self.database.target]
        columns= np.append(self.database.feature_names, ["target"])
        dataset=pd.DataFrame(presented_data, columns=columns)
        print(f"This is the {self.name} dataset: \n", dataset)
  

# create_model method split the data into training and testing parts. Then for each part the data is standardized (it is a must if it is necessary to use 
#MLPClassifier model. Then model is fited by training data- the model is created. As a result this method presents presicion of each model ( in range 0.0-1.0) and
# computes how many mistakes each model makes

    def create_model(self):
        global mlp
        
#spliting data
        X_train, X_test, y_train, y_test= train_test_split(
        self.data, self.database.target, random_state=0
        )

#standardizing data
        mean_on_train= X_train.mean(axis=0)
        std_on_train= X_train.std(axis=0)
        X_train_scaled= (X_train- mean_on_train)/ std_on_train
        X_test_scaled= (X_test- mean_on_train)/ std_on_train

#defining and fitting neural network model        
        mlp=MLPClassifier(max_iter=1000, alpha=1, random_state=0)
        mlp.fit(X_train_scaled, y_train)

#presenting the results
        print("\n")
        print("Used learning model: Neural network")
        print("Precision of the model made of {}  data using training data: {}".format(self.name, mlp.score(X_train_scaled, y_train)))
        print("Precision of the model made of {}  data using testing data: {}".format(self.name, mlp.score(X_test_scaled, y_test)))
        print("")
        quality=mlp.score(X_test_scaled, y_test)
        
#143.0 comes from a shape of testing data
        mistakes=round(143.0*(1.0-quality), 2)
        print("----------------")
        print("The model made of {} data has mistaken {} razy.".format(self.name, mistakes))

        
#plot_heat_map method presents the heat map of input features        
    def plot_heat_map(self):
        plt.figure(figsize=(20,5))
        plt.imshow(mlp.coefs_[0], interpolation='none', cmap='viridis')
        plt.title("{} data".format(self.name))
        plt.yticks(range(30), self.database.feature_names)
        plt.xlabel("Columns of wages' matrices")
        plt.ylabel("Input feature")
        plt.colorbar()
        plt.show()
        
#main is a function in which were used all methods for each data.
def main():
    D1=Model("original", data_cancer, data_cancer.data)
#    D1.print_data()
    D1.create_model()
    D1.plot_heat_map()
    D2=Model("(normal distibution) synthetic", data_cancer, data_cancer.data + 2*(np.random.standard_normal(size=(data_cancer.data.shape))))
#    D2.print_data()
    D2.create_model()
    D2.plot_heat_map()
    D3=Model("(poisson distibution) synthetic", data_cancer, data_cancer.data + np.random.poisson(lam=(.50),size=(data_cancer.data.shape)))
#    D3.print_data()
    D3.create_model()
    D3.plot_heat_map()
    D4=Model("(cauchy distiburion) synthetic", data_cancer, data_cancer.data + np.random.standard_cauchy(size=(data_cancer.data.shape)))
#    D4.print_data()
    D4.create_model()
    D4.plot_heat_map()
#print_data method is commented to prevent ruining the esthetics. I suggest to not use all methods simultaneously. I encourage to try only print_data() method for each 
#object or to combine create_model() method with plot_heat_map() method. 

#calling the main function
if __name__=="__main__":
    main()
