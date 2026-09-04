import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        predictions = []
        for arr in X:
            predictions.append(np.dot(arr, weights))
        # Round to 5 decimal places
        return np.round(predictions, 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        sum = 0
        n = model_prediction.size
        for i in range(n):
            sum += (model_prediction[i] - ground_truth[i]) ** 2
        # Round to 5 decimal places
        return np.round(sum/n, 5).item()
