import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from typing import List, Union, Any, Tuple, Dict
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
# Assuming BaseModel is in the same directory or a part of your package
from .BaseModel import BaseModel
import matplotlib.pyplot as plt
import seaborn as sns


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
        self.feature_names = None # To store feature names for better plots


    def fit(self, X: Union[pd.DataFrame, np.ndarray], y: Any = None) -> None:
        """
        Trains the KPrototypes model.

        Args:
            X: Training data (features).
            y: Ignored (unsupervised learning).
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

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
        # Check if the model has been fitted and if the number of features matches
        if hasattr(self.model, 'cluster_centroids_'):
            num_numerical_features = self.model.cluster_centroids_[0].shape[1]
            num_categorical_features = self.model.cluster_centroids_[1].shape[1]
            # The total number of features used for training is the sum of numerical and categorical
            total_trained_features = num_numerical_features + num_categorical_features
            if X_array.shape[1] != total_trained_features:
                 raise ValueError(f"Number of columns in X ({X_array.shape[1]}) must match the number of features the model was trained on ({total_trained_features}).")
        else:
            # If model not fitted, it's safer to raise an error or handle accordingly
            raise RuntimeError("Model has not been fitted yet. Call fit() before predict().")

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
            except RuntimeError as e: # Catch the RuntimeError from predict if model not fitted
                print(f"Warning: Model not fitted, cannot predict for evaluation. Error: {e}")

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

    def plot_clusters(self, X: Union[pd.DataFrame, np.ndarray], title: str = "K-Prototypes Clustering") -> None:
        """
        Visualizes the clusters. This method provides a basic visualization
        by plotting numerical features and using hue for clusters. For categorical
        features, it prints value counts per cluster.

        Args:
            X: The data used for clustering (features).
            title: Title of the plot.
        """
        if not hasattr(self.model, 'labels_'):
            raise RuntimeError("Model has not been fitted yet. Call fit() before plotting.")

        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=self.feature_names)
        cluster_labels = self.model.labels_
        X_df['cluster'] = cluster_labels

        numerical_features = [col for i, col in enumerate(X_df.columns) if i not in self.categorical_indices and col != 'cluster']
        categorical_features = [col for i, col in enumerate(X_df.columns) if i in self.categorical_indices]

        # --- Numerical Features Visualization ---
        if numerical_features:
            print(f"\n--- Visualizing Numerical Features for {title} ---")
            num_numerical_plots = len(numerical_features)
            # Determine grid size for subplots
            cols = 3 # Max 3 columns per row
            rows = (num_numerical_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
            axes = axes.flatten() # Flatten for easy iteration

            for i, feature in enumerate(numerical_features):
                sns.boxplot(x='cluster', y=feature, data=X_df, ax=axes[i], palette='viridis')
                axes[i].set_title(f'{feature} by Cluster')
                axes[i].set_xlabel('Cluster')
                axes[i].set_ylabel(feature)

            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.suptitle(f"Distribution of Numerical Features Across Clusters\n{title}", y=1.02, fontsize=16)
            plt.show()

            # Pair plot for first few numerical features if many
            if len(numerical_features) >= 2:
                print("\n--- Pair Plot of First 2 Numerical Features (if applicable) ---")
                try:
                    sns.pairplot(X_df, vars=numerical_features[:2], hue='cluster', palette='viridis', diag_kind='kde')
                    plt.suptitle(f"Pair Plot of Numerical Features by Cluster\n{title}", y=1.02, fontsize=16)
                    plt.show()
                except Exception as e:
                    print(f"Could not generate pair plot (might need more than one numerical feature or data issues): {e}")


        # --- Categorical Features Visualization ---
        if categorical_features:
            print(f"\n--- Visualizing Categorical Features for {title} ---")
            num_categorical_plots = len(categorical_features)
            # Determine grid size for subplots
            cols = 2 # Max 2 columns per row for categorical
            rows = (num_categorical_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            axes = axes.flatten()

            for i, feature in enumerate(categorical_features):
                # Count plot to show distribution of categories within each cluster
                sns.countplot(x=feature, hue='cluster', data=X_df, ax=axes[i], palette='viridis')
                axes[i].set_title(f'{feature} Distribution by Cluster')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Count')
                axes[i].tick_params(axis='x', rotation=45) # Rotate labels if needed

            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.suptitle(f"Distribution of Categorical Features Across Clusters\n{title}", y=1.02, fontsize=16)
            plt.show()

        # --- Centroid Visualization (more abstract) ---
        print("\n--- Cluster Centroids ---")
        numerical_centroids, categorical_centroids = self.get_centroids()
        if numerical_centroids is not None and categorical_centroids is not None:
            print("\nNumerical Centroids:")
            num_cols = [self.feature_names[i] for i in range(len(self.feature_names)) if i not in self.categorical_indices]
            print(pd.DataFrame(numerical_centroids, columns=num_cols, index=[f'Cluster {i}' for i in range(self.n_clusters)]))

            print("\nCategorical Centroids:")
            cat_cols = [self.feature_names[i] for i in self.categorical_indices]
            # Convert categorical centroids to a more readable format (e.g., actual categories if possible)
            # For simplicity, here we just print the raw categorical centroids from kmodes
            # which are internal representations.
            print(pd.DataFrame(categorical_centroids, columns=cat_cols, index=[f'Cluster {i}' for i in range(self.n_clusters)]))
        else:
            print("Centroids are not available. Model might not be fitted.")

        print("\n--- Cluster Sizes ---")
        print(X_df['cluster'].value_counts().sort_index())