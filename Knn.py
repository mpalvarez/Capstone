import numpy as np
from collections import Counter

class KNN(object):
    
    def __init__(self, k = 3):
        self.k = k
        pass

    def train(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        pass

    def distances(self, X):
        distances = []
        for i in X:
            for j in self.X_train:
                dist = np.sqrt((i-j)**2)
                distances = [(X, distances)]
    
        return dist