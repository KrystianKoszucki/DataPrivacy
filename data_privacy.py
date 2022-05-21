#program służący do ochrony wykorzystywanej bazy danych i utworzonego dzięki niej modelu

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import pandas as pd

data=load_breast_cancer()
a=data.data
#print(a.shape)
normal_noise=2*(np.random.standard_normal(size=(569, 30)))
noise_data= a + normal_noise
#print(noise_data.shape)

real_data=np.c_[a, data.target]
columns= np.append(data.feature_names, ["target"])
real_dataset=pd.DataFrame(real_data, columns=columns)
synth_data=np.c_[noise_data, data.target]
synth_datases=pd.DataFrame(synth_data, columns=columns)

print("Oryginalne dane (wycinek): \n", real_dataset)
print("\n")
print("Zaszumione dane (wycinek): \n", synth_datases)

X_train, X_test, y_train, y_test= train_test_split(
    a, data.target, random_state=0
)

mean_on_train= X_train.mean(axis=0)
std_on_train= X_train.std(axis=0)
X_train_scaled= (X_train- mean_on_train)/ std_on_train
X_test_scaled= (X_test- mean_on_train)/ std_on_train


mlp= MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp.fit(X_train_scaled, y_train)


X_train_syn, X_test_syn, y_train_syn, y_test_syn= train_test_split(
    noise_data, data.target, random_state=0
)

mean_on_train_syn= X_train_syn.mean(axis=0)
std_on_train_syn= X_train_syn.std(axis=0)
X_train_scaled_syn= (X_train_syn- mean_on_train_syn)/ std_on_train_syn
X_test_scaled_syn= (X_test_syn- mean_on_train_syn)/ std_on_train_syn

mlp_syn=MLPClassifier(max_iter=1000, alpha=1, random_state=0)
mlp_syn.fit(X_train_scaled_syn, y_train_syn)

print("---------------------")
print("\n")
print("Wykorzystany model uczenia: Sieci neuronowe")
print("Dokładność na oryginalnych danych w zestawie uczącym wynosi: {}".format(mlp.score(X_train_scaled, y_train)))
print("Dokładność na oryginalnych danych w zestawie testowym wynosi: {}".format(mlp.score(X_test_scaled, y_test)))
print("\n")
print("\n")
print("Dokładność na syntetycznych danych w zestawie uczącym wynosi: {}".format(mlp_syn.score(X_train_scaled_syn, y_train_syn)))
print("Dokładność na syntetycznych danych w zestawie testowym wynosi: {}".format(mlp_syn.score(X_test_scaled_syn, y_test_syn)))

#print("Kształt oryginalnych danych testowych X_train_scaled: {}".format(X_train_scaled.shape))
#print("Kształt oryginalnych danych testowych X_train_scaled: {}".format(y_train.shape))
#print("\n")
#print("Kształt oryginalnych danych testowych X_train_scaled: {}".format(X_test_scaled.shape))
#print("Kształt oryginalnych danych testowych X_train_scaled: {}".format(y_test.shape))
#print("\n")
#print(y_test.shape)

quality=mlp.score(X_test_scaled, y_test)
quality_syn=mlp_syn.score(X_test_scaled_syn, y_test_syn)


mistakes=round(143.0*(1.0-quality), 2)
mistakes_syn=round(143.0*(1.0-quality_syn), 2)

print("----------------")
print("Model nauczony oryginalnymi danymi pomylił się {} razy.".format(mistakes))
print("Model nauczony syntetycznymi danymi pomylił się {} razy.".format(mistakes_syn))
print("\n")


def plot_feature_importances(model):
    plt.figure(figsize=(20,5))
    plt.imshow(model.coefs_[0], interpolation='none', cmap='viridis')
    if model==mlp:
        plt.title("Realne dane")
    elif model==mlp_syn:
        plt.title("Dane syntetyczne")
    else:
        plt.title("")
    plt.yticks(range(30), data.feature_names)
    plt.xlabel("Kolumny w macierzy wag")
    plt.ylabel("Cecha wejściowa")
    plt.colorbar()
    plt.show()

plot_feature_importances(mlp)
plot_feature_importances(mlp_syn)
