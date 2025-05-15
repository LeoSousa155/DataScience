import unittest
from collections import Counter

import numpy as np
from .BaseModel import BaseModel
from .BallTree import BallTree


class KNN(BaseModel):
    """
    K-Nearest Neighbors classifier/regressor using a custom BallTree
    for efficient neighbor search.
    """

    def __init__(self, k=3, problem_type='classification', leaf_size=10):
        """
        Initialize the KNN model.

        Args:
            k (int): The number of nearest neighbors to consider for prediction.
                     Defaults to 3.
            problem_type (str): Specifies the type of problem.
                                 Must be one of:
                                 - 'classification': For discrete target variables.
                                 - 'regression': For continuous target variables.
                                 Defaults to 'classification'.
            leaf_size (int): The number of points at which a leaf node is created
                             in the BallTree. Defaults to 10.
        """
        self.k = k
        self.ball_tree = BallTree(leaf_size=leaf_size)
        self.y_train = None
        self.is_regression = None
        self.problem_type = problem_type.lower()

        if self.problem_type not in {'classification', 'regression'}:
            raise ValueError('problem type must be either "classification" or "regression"')

    def fit(self, X, y):
        """
        Fit the KNN model to the training data.
        """
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data cannot be empty.")

        self.ball_tree.fit(X, y)
        self.y_train = np.array(y)

        if self.problem_type == 'classification':
            self.is_regression = False
        else:  # regression
            self.is_regression = True

    def predict(self, X):
        """
        Predict the target values for the given test data.
        """
        if self.y_train is None or len(self.y_train) == 0:
            raise ValueError("Model has not been trained with any data.")

        X = np.array(X, dtype=np.float64)
        k_actual = min(self.k, len(self.y_train))
        distances, neighbor_indices = self.ball_tree.query(X, k=k_actual)

        predictions = []
        for i, indices in enumerate(neighbor_indices):
            neighbors_y = self.y_train[indices]

            if len(neighbors_y) == 0:
                predictions.append(0 if not self.is_regression else 0.0)
                continue

            if self.is_regression:
                predictions.append(np.mean(neighbors_y))
            else:
                # Get counts of each class
                counts = Counter(neighbors_y)
                max_count = max(counts.values())
                # Get all classes with max count
                candidates = [k for k, v in counts.items() if v == max_count]
                # Sort to ensure consistent selection (choose smallest class label)
                predictions.append(min(candidates))

        return np.array(predictions)





class TestKNN(unittest.TestCase):
    """
    Unit tests for the KNN class.
    """

    def test_knn_classification(self):
        """
        Test KNN classification with a simple dataset.
        """
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[4, 5], [2, 3]], dtype=np.float64)
        knn = KNN(k=3, problem_type='classification')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        # First test point [4,5] could be either 0 or 1 depending on third neighbor
        self.assertTrue(y_pred[0] in [0, 1])
        # Second test point [2,3] should definitely be 0
        self.assertEqual(y_pred[1], 0)

    def test_knn_regression(self):
        """
        Test KNN regression with a simple dataset.
        """
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y_train = np.array([10, 20, 30, 40], dtype=np.float64)
        X_test = np.array([[4, 5], [6, 7]], dtype=np.float64)
        knn = KNN(k=2, problem_type='regression')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        expected_predictions = np.array([25, 35], dtype=np.float64)  # (20+30)/2, (30+40)/2
        self.assertTrue(np.allclose(y_pred, expected_predictions))

    def test_knn_classification_k1(self):
        """
        Test KNN classification with k=1.
        """
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[4, 5]], dtype=np.float64)
        knn = KNN(k=1, problem_type='classification')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # With k=1, nearest neighbor is [3,4] (label 0) or [5,6] (label 1)
        # We need to accept either since distances are equal
        self.assertTrue(y_pred[0] in [0, 1])

    def test_knn_regression_k1(self):
        """
        Test KNN regression with k=1.
        """
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y_train = np.array([10, 20, 30, 40], dtype=np.float64)
        X_test = np.array([[4, 5]], dtype=np.float64)
        knn = KNN(k=1, problem_type='regression')
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # With k=1, could be either 20 or 30 since equidistant
        self.assertTrue(y_pred[0] in [20.0, 30.0])

    def test_knn_different_leaf_size(self):
        """
        Test KNN with a different leaf size.
        """
        X_train = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=np.float64)
        y_train = np.array([0, 0, 1, 1])
        X_test = np.array([[4, 5]], dtype=np.float64)
        knn = KNN(k=1, problem_type='classification', leaf_size=1)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        self.assertTrue(y_pred[0] in [0, 1])

    def test_knn_invalid_problem_type(self):
        """
        Test KNN with an invalid problem_type.
        """
        with self.assertRaises(ValueError):
            KNN(problem_type='invalid_type')

    def test_knn_empty_training_data(self):
        """
        Test KNN with empty training data.
        """
        X_train = np.empty((0, 2), dtype=np.float64)
        y_train = np.empty((0,), dtype=np.int64)
        knn = KNN(k=1, problem_type='classification')
        with self.assertRaises(ValueError):
            knn.fit(X_train, y_train)

    def test_knn_k_greater_than_samples(self):
        """
        Test KNN when k is greater than the number of training samples.
        """
        X_train = np.array([[1, 2], [3, 4]], dtype=np.float64)
        y_train = np.array([0, 1])
        X_test = np.array([[2, 3]], dtype=np.float64)
        knn = KNN(k=3, problem_type='classification')  # k > n_samples
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        # With k=3 but only 2 samples, should use both
        # Labels are 0 and 1, tie is broken by choosing smaller label (0)
        self.assertEqual(y_pred[0], 0)

if __name__ == '__main__':
    unittest.main()