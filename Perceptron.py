import numpy as np

class Perceptron(object):
    """
    PARAMETRY
    eat - zmiennoprzecinkowy wspolczynnik uczenia miedzy 0 a 1 
    n_iter - liczba calkowita , liczba przebiegow po zestawach uczacych
    random_state: liczba calkowita, ziarno generatora liczb losowych slużące do inicjowania losowych wag

    ATRYBUTY
    w_ - jenowymiarowa tablica wag dopasowania
    errors_ lista, liczba nieprawidlowych klasyfikacji w kazdej epoce


    PARAMETER
    EAT - Gleitkomma-Lernkoeffizient zwischen 0 und 1 
    n_iter - Gesamtzahl der Durchgänge, Anzahl der Durchläufe nach Lernsätzen
    random_state: Gesamtzahl, Zufallszahlengeneratorkorn, das zum Initiieren von Zufallsgewichten verwendet wird

    ATTRIBUTE
    w_ - eindimensionale Anordnung von Einstellgewichten
    errors_ Liste die Anzahl der falschen Klassifikationen in jeder Epoche
    """
    def __init__(self, eta = 0.01, n_iter = 50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state=random_state
    
    def fit(self,X,y):
        """
        Dopasowanie danych uczacych

        parametry
        X - tablicopodobny wymiary =[n_probek, n_cech]

        Abgleich von Schülerdaten

        Parameter
        X - arrayartige Abmessungen =[n_ von Beispielen, n_ Merkmalen]

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale =0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range (self.n_iter):
            errors = 0
            for xi, target in zip (X,y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:]+= update*xi
                self.w_[0]+=update
                errors += int(update!= 0.0)
                
            self.errors_.append(errors)
            

        return self

    def net_input(self, X):
        """oblicza calkowite pobudzenie
        berechnet die totale Erregung"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        #Zwraca etykiete klas po obliczeniu funkcji skoku jednostokowego
        return np.where(self.net_input(X) >= 0.0, 1, -1)