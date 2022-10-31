import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from Perceptron import Perceptron
from matplotlib.colors import ListedColormap

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

class AdalineGD(object):
    """
    PARAMETRY
    eat - zmiennoprzecinkowy wspolczynnik uczenia miedzy 0 a 1 
    n_iter - liczba calkowita , liczba przebiegow po zestawach uczacych
    random_state: liczba calkowita, ziarno generatora liczb losowych slużące do inicjowania losowych wag

    ATRYBUTY
    w_ - jenowymiarowa tablica wag dopasowania
    cost_ lista, suma kwadratów błedów (wartość funkcji kosztu) w każdej epoce

    PARAMETER
    EAT - Gleitkomma-Lernkoeffizient zwischen 0 und 1 
    n_iter - Gesamtzahl der Durchgänge, Anzahl der Durchläufe nach Lernsätzen
    random_state: Gesamtzahl, Zufallszahlengeneratorkorn, das zum Initiieren von Zufallsgewichten verwendet wird

    ATTRIBUTE
    w_ - eindimensionale Anordnung von Einstellgewichten
    cost_ Liste die Summe der Fehlerquadrate (der Wert der Kostenfunktion) in jeder Epoche
    """
    def __init__(self, eta = 0.01, n_iter = 50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state=random_state
    
    def fit(self,X,y):
        """
        Trenowanie za pomocą danych uczących

        parametry
        X - tablicopodobny wymiary =[n_probek, n_cech]

        y - tablicopodobny - wartości docelowe



        Trainieren mit Lerndaten

    Parameter
        X - arrayartige Dimensionen =[n_probek, n_cech]

        y - arrayartig - Zielwerte

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale =0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range (self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:]+= self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum()/2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """oblicza calkowite pobudzenie
        
        berechnet die totale Erregung"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Oblicza liniową funkcje aktyacji
        
        Berechnet lineare Aktationsfunktion"""
        return X

    def predict(self, X):
        #Zwraca etykiete klas po obliczeniu funkcji skoku jednostokowego   Gibt die Klassenbezeichnung zurück, nachdem die Einzelaugensprungfunktion berechnet wurde
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, -1)
"""
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=(10,4))
ada1 = AdalineGD(n_iter=10, eta=0.01).fit(X,y)
ax[0].plot(range(1,len(ada1.cost_)+1),
            np.log10(ada1.cost_),marker = 'o')
ax[0].set_xlabel("Epoki")
ax[0].set_xlabel("Log(suma kwadratów błędów)")
ax[0].set_title("Adaline współczynnik uczenia 0.01")
ada2 = AdalineGD(n_iter=10, eta = 0.0001).fit(X,y)
ax[1].plot(range(1, len(ada2.cost_)+1), ada2.cost_, marker = 'o')
ax[1].set_xlabel("Epoki")
ax[1].set_xlabel("Suma kwadratów błędów")
ax[1].set_title("Adaline - Współczynnik uczenia 0.0001")
plt.show()"""

X_std = np.copy(X)
X_std[:,0] = (X[:,0] - X[:,0].mean())/ X[:,0].std()
X_std[:,1] = (X[:,1] - X[:,1].mean())/ X[:,1].std()

ada = AdalineGD(eta=0.01, n_iter=15)
ada.fit(X_std, y)
plot_decision_regions(X_std, y, classifier=ada)
plt.title("Adaline - Gradient prosty")
plt.xlabel("Dlugosc dzialki [standaryzowana]")
plt.ylabel("dlugosc platka [standaryzowana]]")
plt.legend(loc ='upper left')
plt.tight_layout()
plt.show()

plt.plot(range(1,len(ada.cost_)+1), ada.cost_,marker='o')
plt.xlabel("Epoki")
plt.ylabel("Suma kwadratów błędów") #Summe der Fehlerquadrate
plt.show()