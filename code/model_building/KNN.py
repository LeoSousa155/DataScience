import BaseModel

import numpy as np
from collections import Counter

class KNN(BaseModel):
    def __init__(self, k=3, problem_type='auto'):
        """
        Initialize the KNN model.

        Args:
            k (int): Number of neighbors to consider.
            problem_type (str): 'auto' (detect automatically), 'regression', or 'classification'.
        """
        self.k = k
        self.X_train = None
        self.y_train = None
        self.is_regression = None
        self.problem_type = problem_type.lower()  # Store the problem type

    def fit(self, X, y):
        self.X_train = np.array(X, dtype=np.float64)
        self.y_train = np.array(y)

        # Determine problem type
        if self.problem_type == 'auto':
            unique_values = np.unique(self.y_train)
            self.is_regression = len(unique_values) > 10 or isinstance(self.y_train[0], float)
        elif self.problem_type == 'regression':
            self.is_regression = True
        elif self.problem_type == 'classification':
            self.is_regression = False
        else:
            raise ValueError("problem_type must be 'auto', 'regression', or 'classification'")

        if not self.is_regression:
            print("Mode: Classification (discrete target)")
        else:
            print("Mode: Regression (continuous target)")

    def predict(self, X):
        X = np.array(X, dtype=np.float64)
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        distances = np.linalg.norm(self.X_train - x, axis=1)
        k_nearest_indices = np.argpartition(distances, self.k)[:self.k]
        k_nearest_values = self.y_train[k_nearest_indices]

        if self.is_regression:
            # For regression: return mean of k nearest values
            return np.mean(k_nearest_values)
        else:
            # For classification: return most common class
            most_common = Counter(k_nearest_values).most_common(1)[0][0]
            return most_common