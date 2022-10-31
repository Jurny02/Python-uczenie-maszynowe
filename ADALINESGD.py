import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import Perceptron
from matplotlib.colors import ListedColormap
from numpy.random import seed

import numpy as np

from irys_test_Perceptron import plot_decision_regions
df = pd.read_csv('https://archive.ics.uci.edu/ml/'
'machine-learning-databases/iris/iris.data', header=None)
print(df.tail())

#wybieramy odmiany setosa i versicolor
y = df.iloc[0:100,4].values
y = np.where(y == 'Iris-setosa', -1, 1)

#wybieramy dlugosc dzialki i dlugosc platka
X = df.iloc[0:100, [0,2]].values

class AdalineSGD(object):
    """
    PARAMETRY
    eat - zmiennoprzecinkowy wspolczynnik uczenia miedzy 0 a 1 
    n_iter - liczba calkowita , liczba przebiegow po zestawach uczacych
    shuffle: wartosc boolowska (domyslne True)
        jesli True - tasuje dane uczące przez każdą epoką w celu zapobiegnięcia cykliczności
    random_state: liczba calkowita, ziarno generatora liczb losowych slużące do inicjowania losowych wag

    ATRYBUTY
    w_ - jenowymiarowa tablica wag dopasowania
    cost_ lista, suma kwadratów błedów (wartość funkcji kosztu) w każdej epoce
    """
    def __init__(self, eta = 0.01, n_iter = 50, shuffle=True, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.w_initialized = False
        self.shuffle = shuffle
        self.random_state=random_state
    
    def fit(self,X,y):
        """
        Dopasowanie danych uczących

        parametry
        X - tablicopodobny wymiary =[n_probek, n_cech]

        y - tablicopodobny - wartości docelowe

        zwaraca obiekt self
        """
        self.initialize_weights(X.shape[1])
        self.cost_ = []
        for i in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X,y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost)/ len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        #Dopasowanie dane uczące bez ponownej inicjacji wag
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip (X,y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X,y)
        return self
    
    def shuffle(self,X, y):
        #Tasuje dane uczące
        r = self.rgen.permutation(len(y))
        return X[r], y[r]
    
    def _initilize_weights(self, m):
        #Inicjuje wagi przydzielając im małe, losoe wartości
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size = 1 + m)
        self.w_initialized = True
    
    def _update_weights(self, xi, target):
        #Wykorzystuje regółę uczenia Adaline do akuralizacji wag
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0]+=self.eta*error
        cost = 0.5*error**2
        return cost

    def net_input(self, X):
        """oblicza calkowite pobudzenie"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Oblicza liniową funkcje aktyacji"""
        return X

    def predict(self, X):
        #Zwraca etykiete klas po obliczeniu funkcji skoku jednostokowego
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)