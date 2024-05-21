import numpy as np
from tqdm import tqdm

from collections import Counter


class KNN:
    def __init__(self, k:int=3):
        self.k = k

    def fit(self, X:np.ndarray, y:np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X_pred, batch_size=128):
        pred = []
        for i in tqdm(range(0, len(X_pred), batch_size)):
            X = X_pred[i:min(len(X_pred), i+batch_size)]
            distances = np.sqrt(np.sum((X[:, np.newaxis] - self.X_train)**2, axis=2))
            k_indices = np.argsort(distances, axis=1)[:, :self.k]
            k_nearest_labels = self.y_train[k_indices]
            y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=k_nearest_labels)
            pred.extend(y_pred.tolist())
        return np.array(pred)
