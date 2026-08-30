import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        L = 0
        for i in range(y_true.size):
            y = y_true[i]
            p = y_pred[i] + (1e-7)
            L += y * np.log(p) + (1 - y) * np.log(1 - p)
        L *= -1/y_true.size
        return round(L, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        L = 0
        for i in range(y_true.shape[0]):
            C = y_true[i]
            for c in range(C.size):
                y = y_true[i][c]
                p = y_pred[i][c]
                L += y * np.log(p  + (1e-7))
        L *= -1/y_true.shape[0]
        return round(L, 4)
