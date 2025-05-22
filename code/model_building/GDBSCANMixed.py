import pandas as pd
import numpy as np
from typing import Union, Any, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from .BaseModel import BaseModel
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


class GDBSCANMixed(BaseModel):
    """
    A density-based clustering algorithm that can handle mixed numerical and
    categorical data by using a custom distance metric (defaulting to Gower distance).

    GDBSCAN (Generalized DBSCAN) extends the DBSCAN algorithm to work with
    non-Euclidean distance metrics, making it suitable for data with mixed types.
    It groups together data points that are closely packed together (points with
    many nearby neighbors), marking as outliers points that lie alone in low-density
    regions.
    """
    def __init__(self, eps: float, min_samples: int, distance_metric: Union[str, callable] = 'gower'):
        """
        Initializes the GDBSCANMixed clustering model.

        Args:
            eps (float): The maximum distance between two samples for one to be
                considered as in the neighborhood of the other. This is a crucial
                parameter that influences the size and density of the clusters.
            min_samples (int): The number of samples in a neighborhood for a point
                to be considered as a core point. This parameter controls the
                minimum density required to form a cluster.
            distance_metric (str or callable, optional): The metric to use when
                calculating the distance between data points.
                - If 'gower', a conceptual Gower distance calculation is used
                  (it's recommended to use a proper implementation from a library
                  like 'gower' for real-world applications).
                - If a callable, it should be a function that takes the data as
                  input and returns a pairwise distance matrix. The function
                  should be able to handle the mixed data types in your dataset.
                Defaults to 'gower'.
        """
        if not isinstance(eps, (int, float)) or eps <= 0:
            raise ValueError("eps must be a positive float.")
        if not isinstance(min_samples, int) or min_samples <= 0:
            raise ValueError("min_samples must be a positive integer.")
        if not (isinstance(distance_metric, str) and distance_metric == 'gower') and not callable(distance_metric):
            raise ValueError("distance_metric must be 'gower' or a callable.")

        self.eps = eps
        self.min_samples = min_samples
        self.distance_metric = distance_metric
        self.labels = None
        self._train_data = None # Store training data for prediction
        self.feature_names = None # To store feature names for better plots

    def _gower_distance(self, data: pd.DataFrame) -> np.ndarray:
        """
        Calculates the Gower distance between all pairs of data points in the
        input DataFrame. This metric is suitable for mixed numerical and
        categorical data.

        Note: This is a conceptual implementation. For accurate and efficient
        Gower distance calculation, it is highly recommended to use a dedicated
        library like 'gower'.

        Args:
            data (pd.DataFrame): The input DataFrame containing mixed numerical
                and categorical features.

        Returns:
            np.ndarray: A square matrix where the (i, j)-th element is the
            Gower distance between the i-th and j-th data points.
        """
        n_samples, n_features = data.shape
        distance_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i, n_samples):
                dist = 0
                for k in range(n_features):
                    val_i = data.iloc[i, k]
                    val_j = data.iloc[j, k]
                    series_k = data.iloc[:, k]
                    if pd.api.types.is_numeric_dtype(series_k):
                        range_k = series_k.max() - series_k.min()
                        if range_k > 0:
                            dist += abs(val_i - val_j) / range_k
                        else:
                            dist += 0 # If range is 0, treat as identical
                    else:
                        if val_i != val_j:
                            dist += 1
                distance_matrix[i, j] = distance_matrix[j, i] = dist / n_features
        return distance_matrix

    def fit(self, X: pd.DataFrame) -> None:
        """
        Performs GDBSCAN clustering on the input data.

        Args:
            X (pd.DataFrame): The input DataFrame containing the data to cluster.
                It should contain both numerical and categorical features.

        Returns:
            None: The cluster labels are stored in the `self.labels` attribute.
        """
        self._train_data = X.copy() # Store training data for prediction
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        n_samples = X.shape[0]
        self.labels = np.full(n_samples, -1, dtype=int) # Initialize all points as noise (-1)
        cluster_id = 0

        if self.distance_metric == 'gower':
            distance_matrix = self._gower_distance(X)
        else:
            # Assume a callable custom distance metric is provided
            distance_matrix = self.distance_metric(X)

        for i in range(n_samples):
            if self.labels[i] != -1:
                continue

            # Find neighbors within eps distance of the current point
            neighbors_indices = np.where(distance_matrix[i] <= self.eps)[0]

            if len(neighbors_indices) < self.min_samples:
                # If the number of neighbors is less than min_samples, mark as noise
                self.labels[i] = -1
            else:
                # If enough neighbors, start a new cluster and expand it
                self._expand_cluster(X, i, neighbors_indices, cluster_id, distance_matrix)
                cluster_id += 1

    def _expand_cluster(self, X: pd.DataFrame, start_point: int, neighbors_indices: np.ndarray,
                         cluster_id: int, distance_matrix: np.ndarray) -> None:
        """
        Recursively expands a cluster starting from a core point.

        Args:
            X (pd.DataFrame): The input DataFrame.
            start_point (int): The index of the core point from which to start
                the cluster expansion.
            neighbors_indices (np.ndarray): Array of indices of the neighbors
                of the start point within eps.
            cluster_id (int): The ID to assign to the current cluster.
            distance_matrix (np.ndarray): The pairwise distance matrix.

        Returns:
            None: The `self.labels` attribute is updated with the cluster assignments.
        """
        self.labels[start_point] = cluster_id
        queue = list(neighbors_indices)
        i = 0
        while i < len(queue):
            current_point_index = queue[i]
            if self.labels[current_point_index] == -1:
                # Assign the current point to the cluster
                self.labels[current_point_index] = cluster_id
                # Find neighbors of the current point
                new_neighbors_indices = np.where(distance_matrix[current_point_index] <= self.eps)[0]
                # If the current point is a core point (has enough neighbors),
                # add its neighbors to the queue to expand the cluster further
                if len(new_neighbors_indices) >= self.min_samples:
                    for neighbor_index in new_neighbors_indices:
                        # Only add if not already processed and not yet assigned to a cluster
                        if self.labels[neighbor_index] == -1: # Check for -1 (noise or unvisited)
                            queue.append(neighbor_index)
            i += 1

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts cluster labels for new, unseen data points.

        This is a simplified prediction method for density-based clustering.
        It assigns a new point to the cluster of its nearest neighbor in the
        training data if that neighbor is a core point and the distance is
        within eps. Otherwise, the new point is labeled as noise (-1).

        Args:
            X (pd.DataFrame): The new data points to predict cluster labels for.

        Returns:
            np.ndarray: An array of cluster labels for each new data point.
                -1 indicates noise.
        """
        if self.labels is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")
        if self._train_data is None:
            raise ValueError("Training data not stored. Ensure fit() was called.")

        new_labels = np.full(X.shape[0], -1, dtype=int) # Default to noise

        # Concatenate train and test data to calculate distances consistently
        combined_data = pd.concat([self._train_data, X], ignore_index=True)

        if self.distance_metric == 'gower':
            full_distance_matrix = self._gower_distance(combined_data)
        else:
            raise NotImplementedError("Prediction for custom distance metric not implemented for callable distance_metric.")

        # Extract the relevant part of the distance matrix: distances from new points to training points
        # Rows correspond to new points, columns to training points
        distances_new_to_train = full_distance_matrix[len(self._train_data):, :len(self._train_data)]
        
        # Determine core points from the training data
        train_core_points = np.zeros(len(self._train_data), dtype=bool)
        for i in range(len(self._train_data)):
            neighbors_count = np.sum(full_distance_matrix[i, :len(self._train_data)] <= self.eps)
            if neighbors_count >= self.min_samples:
                train_core_points[i] = True

        for i in range(X.shape[0]):
            distances_to_train = distances_new_to_train[i]
            # Find all training points within eps distance
            potential_neighbors_indices = np.where(distances_to_train <= self.eps)[0]

            assigned_cluster = -1
            # Prioritize core points within eps
            for nn_idx in potential_neighbors_indices:
                if train_core_points[nn_idx] and self.labels[nn_idx] != -1: # Ensure it's a core point and part of a cluster
                    assigned_cluster = self.labels[nn_idx]
                    break # Assign to the first found core point's cluster

            new_labels[i] = assigned_cluster

        return new_labels

    def evaluate(self, X_test: Union[pd.DataFrame, None] = None,
                     y_test: Union[np.ndarray, pd.Series, None] = None) -> Dict[str, Any]:
        """
        Evaluates the clustering performance using external evaluation metrics
        if ground truth labels are provided.

        Args:
            X_test (pd.DataFrame, optional): Test data (not used for evaluation
                in unsupervised clustering, included for API consistency).
                Defaults to None.
            y_test (np.ndarray, pd.Series, optional): Ground truth cluster labels
                for the data used in the fit method. Defaults to None.

        Returns:
            dict: A dictionary containing evaluation metrics. Currently includes
            Adjusted Rand Score and Normalized Mutual Information if y_test
            is provided and the number of labels matches.
        """
        results = {}
        if y_test is not None and self.labels is not None and len(self.labels) == len(y_test):
            # Ensure y_test is a numpy array for consistent indexing
            if isinstance(y_test, pd.Series):
                y_test = y_test.values

            # Filter out noise points (-1) from both predicted and true labels for evaluation
            # as these metrics typically don't handle noise labels directly
            non_noise_indices = np.where(self.labels != -1)[0]
            if len(non_noise_indices) > 0:
                filtered_labels = self.labels[non_noise_indices]
                filtered_y_test = y_test[non_noise_indices]

                # Ensure there's more than one unique label for metrics to be meaningful
                if len(np.unique(filtered_labels)) > 1 and len(np.unique(filtered_y_test)) > 1:
                    results['adjusted_rand_score'] = adjusted_rand_score(filtered_y_test, filtered_labels)
                    results['normalized_mutual_info_score'] = normalized_mutual_info_score(filtered_y_test, filtered_labels)
                else:
                    print("Warning: Not enough unique labels after filtering noise for ARI/NMI calculation.")
            else:
                print("Warning: All points labeled as noise, cannot calculate ARI/NMI.")
        return results

    def get_centroids(self) -> None:
        """
        Returns the centroids of the clusters.

        Density-based methods like GDBSCAN do not typically have well-defined
        centroids in the same way as partition-based or model-based methods.
        Returns None.
        """
        return None

    def plot_clusters(self, X: pd.DataFrame, title: str = "GDBSCAN Clustering") -> None:
        """
        Visualizes the clusters formed by GDBSCAN.

        Args:
            X (pd.DataFrame): The data used for clustering (features).
            title (str): Title of the plot.
        """
        if self.labels is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before plotting.")

        X_df = X.copy()
        X_df['cluster'] = self.labels

        # Identify numerical and categorical features
        numerical_features = [col for col in X_df.columns if pd.api.types.is_numeric_dtype(X_df[col]) and col != 'cluster']
        categorical_features = [col for col in X_df.columns if not pd.api.types.is_numeric_dtype(X_df[col]) and col != 'cluster']

        # --- Numerical Features Visualization (Scatter Plots) ---
        if len(numerical_features) >= 2:
            print(f"\n--- Visualizing Numerical Features for {title} ---")
            # Create scatter plots for pairs of numerical features
            # Using hue for 'cluster' and style for distinguishing noise
            
            # Map cluster labels to string for better legend and hue handling
            X_df['cluster_str'] = X_df['cluster'].astype(str)
            # Replace -1 with 'Noise' for plotting
            X_df['cluster_str'] = X_df['cluster_str'].replace('-1', 'Noise')

            # Determine unique clusters for palette
            unique_clusters = sorted(X_df['cluster_str'].unique())
            # Ensure 'Noise' is always last and potentially a distinct color (e.g., grey/black)
            if 'Noise' in unique_clusters:
                unique_clusters.remove('Noise')
                unique_clusters.append('Noise')
            
            # Generate a color palette, ensuring 'Noise' has a specific color if present
            palette = sns.color_palette("viridis", n_colors=len(unique_clusters) - (1 if 'Noise' in unique_clusters else 0))
            if 'Noise' in unique_clusters:
                palette.append('gray') # Assign gray to noise
            cluster_palette = dict(zip(unique_clusters, palette))


            # Plotting only the first two numerical features for simplicity in pairplot
            if len(numerical_features) >= 2:
                plt.figure(figsize=(10, 8))
                sns.scatterplot(
                    x=numerical_features[0],
                    y=numerical_features[1],
                    hue='cluster_str',
                    style='cluster_str', # Use style to differentiate noise points
                    data=X_df,
                    palette=cluster_palette,
                    s=80, # Marker size
                    alpha=0.7 # Transparency
                )
                plt.title(f'Cluster Visualization of {numerical_features[0]} vs {numerical_features[1]}\n{title}')
                plt.xlabel(numerical_features[0])
                plt.ylabel(numerical_features[1])
                plt.legend(title='Cluster')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.show()
            else:
                print("Not enough numerical features (at least 2 required) for scatter plot visualization.")

        # --- Categorical Features Visualization (Count Plots) ---
        if categorical_features:
            print(f"\n--- Visualizing Categorical Features for {title} ---")
            num_categorical_plots = len(categorical_features)
            cols = 2 # Max 2 columns per row for categorical plots
            rows = (num_categorical_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            axes = axes.flatten()

            for i, feature in enumerate(categorical_features):
                sns.countplot(x=feature, hue='cluster_str', data=X_df, ax=axes[i], palette=cluster_palette)
                axes[i].set_title(f'{feature} Distribution by Cluster')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Count')
                axes[i].tick_params(axis='x', rotation=45) # Rotate labels if needed
                axes[i].legend(title='Cluster')

            # Hide unused subplots
            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.suptitle(f"Distribution of Categorical Features Across Clusters\n{title}", y=1.02, fontsize=16)
            plt.show()
        else:
            print("No categorical features to visualize.")

        print("\n--- Cluster Sizes ---")
        # Ensure 'Noise' is explicitly mentioned for -1
        cluster_counts = X_df['cluster_str'].value_counts().sort_index()
        print(cluster_counts)