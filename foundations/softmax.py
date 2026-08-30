import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        z = z - max(z)
        exp = np.exp(z)
        exp_sum = np.sum(exp)
        ans = exp / exp_sum
        return np.round(ans, 4)
