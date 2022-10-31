import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import Perceptron
from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y , classifier, resolution = 0.02):
    #konfiguruje generator znaczników i mapę kolrów
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyjan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #rysuje wykres powierzchni decyzyjnej
    x1_min, x1_max = X[:, 0].min() - 1, X[:,0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:,1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha = 0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    #rysuje wykres próbek
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y==cl, 0], y = X[y == cl, 1], alpha=0.8, c= colors[idx], marker=markers[idx], label = cl, edgecolors='black')

df = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

#wybieramy odmiany setosa i versicolor
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', -1, 1)

#wybieramy dlugosc dzialki i dlugosc platka
X = df.iloc[0:100, [0,2]].values

#generowanie wykresu
plt.scatter(X[:50, 0],X[:50, 1],
color='red', marker='o', label='Setosa')
plt.scatter(X[50:100, 0],X[50:100, 1],
color='blue', marker='x', label='Versicolor')
plt.xlabel("Dlugosc dzialki [cm]") #Länge des Grundstücks
plt.ylabel("dlugosc platka [cm]") #Länge des Tellers
plt.legend(loc ='upper left')
plt.show()

ppn = Perceptron(eta=0.1, n_iter=10)
ppn.fit(X, y)
plt.plot(range(1,len(ppn.errors_)+1), ppn.errors_,marker='o')
plt.xlabel("Epoki")
plt.ylabel('Liczba aktualizacji') #Anzahl der Updates
plt.show()

plot_decision_regions(X, y, classifier=ppn)
plt.xlabel("Dlugosc dzialki [cm]")
plt.ylabel("dlugosc platka [cm]")
plt.legend(loc ='upper left')
plt.show()


