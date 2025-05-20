import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from typing import List, Union, Any, Tuple, Dict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from .BaseModel import BaseModel


class KPrototypesModel(BaseModel):
    """
    A wrapper class for the kmodes KPrototypes algorithm for clustering
    mixed numerical and categorical data.
    """
    def __init__(self, n_clusters: int, categorical_indices: List[int], **kwargs: Any):
        """
        Initializes the KPrototypesModel.

        Args:
            n_clusters: The number of clusters to form.
            categorical_indices: Indices of categorical columns in the input data.
            **kwargs: Additional arguments for the KPrototypes constructor.
        """
        if not isinstance(n_clusters, int) or n_clusters <= 0:
            raise ValueError("n_clusters must be a positive integer.")
        if not isinstance(categorical_indices, list) or not all(isinstance(i, int) for i in categorical_indices):
            raise ValueError("categorical_indices must be a list of integers.")

        self.n_clusters = n_clusters
        self.categorical_indices = sorted(categorical_indices)
        self.model = KPrototypes(n_clusters=n_clusters, **kwargs)
        self.problem_type = 'clustering'

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None) -> None:
        """
        Trains the KPrototypes model.

        Args:
            X: Training data (features).
            y: Ignored (unsupervised learning).
        """
        X_array = self._validate_data(X)
        if X_array.shape[1] <= max(self.categorical_indices) if self.categorical_indices else False:
            raise ValueError("Number of columns in X must be greater than the maximum index in categorical_indices.")
        self.model.fit(X_array, categorical=self.categorical_indices)

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predicts cluster assignments for new data.

        Args:
            X: New data to predict clusters for.

        Returns:
            Array of cluster labels for each data point.
        """
        X_array = self._validate_data(X)
        if hasattr(self.model, 'cluster_centroids_') and X_array.shape[1] != self.model.cluster_centroids_[0].shape[1] + len(self.categorical_indices):
            raise ValueError("Number of columns in X must match the number of features the model was trained on.")
        return self.model.predict(X_array, categorical=self.categorical_indices)

    def _validate_data(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Ensures input data is a NumPy array.
        """
        if isinstance(X, pd.DataFrame):
            return X.values
        elif isinstance(X, np.ndarray):
            return X
        else:
            raise ValueError("Input data X must be a pandas DataFrame or a NumPy array.")

    def evaluate(self, X_test: Union[pd.DataFrame, np.ndarray] = None,
                 y_test: Union[pd.Series, np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluates the clustering performance.

        Args:
            X_test: Optional test data for external evaluation.
            y_test: Optional ground truth labels for external evaluation.

        Returns:
            Dictionary of evaluation metrics.
        """
        results: Dict[str, Any] = {}
        if not hasattr(self.model, 'cost_'):
            print("Warning: Model not fitted or cost_ attribute missing.")
            return results

        results['cost'] = self.model.cost_
        results['numerical_centroids'] = self.model.cluster_centroids_[0].tolist()
        results['categorical_centroids'] = self.model.cluster_centroids_[1].tolist()

        if X_test is not None and y_test is not None:
            try:
                cluster_labels = self.predict(X_test)
                results['adjusted_rand_score'] = adjusted_rand_score(y_test, cluster_labels)
                results['normalized_mutual_info_score'] = normalized_mutual_info_score(y_test, cluster_labels)
            except ValueError as e:
                print(f"Warning: Could not calculate external evaluation metrics. Ensure X_test has the correct number of features. Error: {e}")

        return results

    def get_centroids(self) -> Union[Tuple[List[List[float]], List[List[Any]]], None]:
        """
        Returns the cluster centroids.

        Returns:
            Tuple of numerical and categorical centroids, or None if not fitted.
        """
        if hasattr(self.model, 'cluster_centroids_'):
            return (self.model.cluster_centroids_[0].tolist(), self.model.cluster_centroids_[1].tolist())
        else:
            return None