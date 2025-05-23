import pandas as pd
import numpy as np
from typing import Union, Any, Dict
import matplotlib.pyplot as plt
import seaborn as sns
import hdbscan
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from .BaseModel import BaseModel


class HDBSCANMixed(BaseModel):  # Renomeando a classe para refletir o algoritmo
    """
    A density-based clustering algorithm using HDBSCAN* which can handle mixed
    numerical and categorical data via intelligent pre-processing.

    HDBSCAN* is a robust and efficient algorithm that extends DBSCAN by not
    requiring the 'eps' parameter and being able to find clusters of varying densities.
    It builds a hierarchy of clusters and extracts the most stable ones.
    """

    def __init__(self, min_cluster_size: int = 5, min_samples: Union[int, None] = None,
                 cluster_selection_epsilon: float = 0.0, metric: str = 'euclidean', **hdbscan_kwargs):
        """
        Initializes the HDBSCANMixed clustering model.

        Args:
            min_cluster_size (int): The minimum size of clusters. Smaller values will allow
                more and smaller clusters to be formed, and can be more sensitive to noise.
                Defaults to 5.
            min_samples (int, optional): The number of samples in a neighborhood for a point to
                be considered a core point. This controls the "conservativeness" of the clustering.
                Defaults to None (uses min_cluster_size).
            cluster_selection_epsilon (float): A threshold for cluster merging. Clusters below
                this threshold in the hierarchy will be merged. Defaults to 0.0.
            metric (str): The metric to use when calculating the distance between data points.
                Since HDBSCAN will be applied after numerical transformation (One-Hot Encoding
                + Scaling), 'euclidean' is typically the default. Other metrics like 'cityblock',
                'cosine', 'minkowski' (with p=1 or 2) can also be used. Defaults to 'euclidean'.
            **hdbscan_kwargs: Additional keyword arguments to pass directly to the hdbscan.HDBSCAN constructor.
                Useful for fine-tuning, e.g., 'prediction_data=True' if you plan to predict on new data.
        """
        if not isinstance(min_cluster_size, int) or min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be a positive integer.")
        if min_samples is not None and (not isinstance(min_samples, int) or min_samples <= 0):
            raise ValueError("min_samples must be a positive integer or None.")
        if not isinstance(cluster_selection_epsilon, (int, float)) or cluster_selection_epsilon < 0:
            raise ValueError("cluster_selection_epsilon must be a non-negative float.")

        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.metric = metric
        self.hdbscan_kwargs = hdbscan_kwargs

        self.labels = None
        self._fitted_pipeline = None  # Store the fitted sklearn pipeline
        self.feature_names_original = None

    def _build_preprocessing_pipeline(self, X: pd.DataFrame) -> Pipeline:
        """
        Builds the scikit-learn pipeline for pre-processing mixed data.
        """
        numeric_features = X.select_dtypes(include=np.number).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Create a preprocessor using ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ],
            remainder='passthrough'  # Keep other columns if any, though usually not needed
        )

        # Create the HDBSCAN model
        hdbscan_model = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric=self.metric,
            **self.hdbscan_kwargs
        )

        # Build the full pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('hdbscan', hdbscan_model)
        ])
        return pipeline

    def fit(self, X: pd.DataFrame) -> None:
        """
        Performs HDBSCAN clustering on the input data after appropriate pre-processing.

        Args:
            X (pd.DataFrame): The input DataFrame containing the data to cluster.
                It should contain both numerical and categorical features.

        Returns:
            None: The cluster labels are stored in the `self.labels` attribute.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Input X must be a pandas DataFrame.")

        self.feature_names_original = X.columns.tolist()
        self._fitted_pipeline = self._build_preprocessing_pipeline(X)

        print("Iniciando o pré-processamento e treinamento HDBSCAN...")
        self._fitted_pipeline.fit(X)
        print("Treinamento concluído.")

        self.labels = self._fitted_pipeline.named_steps['hdbscan'].labels_

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predicts cluster labels for new, unseen data points using the fitted HDBSCAN model.
        Note: HDBSCAN's prediction is different from K-Means. It typically assigns new points
        to the cluster of their closest training point if within a certain density, or to noise.
        This requires `prediction_data=True` in the HDBSCAN constructor during fit.

        Args:
            X (pd.DataFrame): The new data points to predict cluster labels for.

        Returns:
            np.ndarray: An array of cluster labels for each new data point.
                -1 indicates noise.
        """
        if self._fitted_pipeline is None:
            raise ValueError("Model not fitted yet. Call 'fit' first.")

        # Ensure that prediction_data=True was set during initialization if you want a robust predict
        if 'prediction_data' not in self.hdbscan_kwargs or not self.hdbscan_kwargs['prediction_data']:
            print("Warning: For robust prediction with HDBSCAN, initialize with prediction_data=True.")
            print("Falling back to a simpler prediction method: transforming data and then calling predict.")
            # For simpler prediction, just transform new data and use the hdbscan predict method
            # This method usually finds the closest existing cluster point and assigns.
            X_transformed = self._fitted_pipeline.named_steps['preprocessor'].transform(X)
            return self._fitted_pipeline.named_steps['hdbscan'].predict(X_transformed)

        # If prediction_data was true, the hdbscan model is ready for robust prediction
        X_transformed = self._fitted_pipeline.named_steps['preprocessor'].transform(X)
        return self._fitted_pipeline.named_steps['hdbscan'].predict(X_transformed)

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
            if isinstance(y_test, pd.Series):
                y_test = y_test.values

            # Filter out noise points (-1) from both predicted and true labels for evaluation
            non_noise_indices = np.where(self.labels != -1)[0]
            if len(non_noise_indices) > 0:
                filtered_labels = self.labels[non_noise_indices]
                filtered_y_test = y_test[non_noise_indices]

                if len(np.unique(filtered_labels)) > 1 and len(np.unique(filtered_y_test)) > 1:
                    results['adjusted_rand_score'] = adjusted_rand_score(filtered_y_test, filtered_labels)
                    results['normalized_mutual_info_score'] = normalized_mutual_info_score(filtered_y_test,
                                                                                           filtered_labels)
                else:
                    print("Warning: Not enough unique labels after filtering noise for ARI/NMI calculation.")
            else:
                print("Warning: All points labeled as noise, cannot calculate ARI/NMI.")
        return results

    def get_centroids(self) -> None:
        """
        Returns the centroids of the clusters.

        Density-based methods like HDBSCAN* do not typically have well-defined
        centroids in the same way as partition-based or model-based methods.
        Returns None.
        """
        return None

    def plot_clusters(self, X: pd.DataFrame, title: str = "HDBSCAN Clustering") -> None:
        """
        Visualizes the clusters formed by HDBSCAN*.

        Args:
            X (pd.DataFrame): The data used for clustering (features).
            title (str): Title of the plot.
        """
        if self.labels is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() before plotting.")

        X_df = X.copy()
        X_df['cluster'] = self.labels

        # Identify numerical and categorical features
        numerical_features = [col for col in X_df.columns if
                              pd.api.types.is_numeric_dtype(X_df[col]) and col != 'cluster']
        categorical_features = [col for col in X_df.columns if
                                not pd.api.types.is_numeric_dtype(X_df[col]) and col != 'cluster']

        # --- Numerical Features Visualization (Scatter Plots) ---
        if len(numerical_features) >= 2:
            print(f"\n--- Visualizing Numerical Features for {title} ---")

            X_df['cluster_str'] = X_df['cluster'].astype(str)
            X_df['cluster_str'] = X_df['cluster_str'].replace('-1', 'Noise')

            unique_clusters = sorted(X_df['cluster_str'].unique())
            if 'Noise' in unique_clusters:
                unique_clusters.remove('Noise')
                unique_clusters.append('Noise')

            palette = sns.color_palette("viridis",
                                        n_colors=len(unique_clusters) - (1 if 'Noise' in unique_clusters else 0))
            if 'Noise' in unique_clusters:
                palette.append('gray')
            cluster_palette = dict(zip(unique_clusters, palette))

            if len(numerical_features) >= 2:
                plt.figure(figsize=(10, 8))
                sns.scatterplot(
                    x=numerical_features[0],
                    y=numerical_features[1],
                    hue='cluster_str',
                    style='cluster_str',
                    data=X_df,
                    palette=cluster_palette,
                    s=80,
                    alpha=0.7
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
            cols = 2
            rows = (num_categorical_plots + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            axes = axes.flatten()

            for i, feature in enumerate(categorical_features):
                sns.countplot(x=feature, hue='cluster_str', data=X_df, ax=axes[i], palette=cluster_palette)
                axes[i].set_title(f'{feature} Distribution by Cluster')
                axes[i].set_xlabel(feature)
                axes[i].set_ylabel('Count')
                axes[i].tick_params(axis='x', rotation=45)
                axes[i].legend(title='Cluster')

            for j in range(i + 1, len(axes)):
                fig.delaxes(axes[j])

            plt.tight_layout()
            plt.suptitle(f"Distribution of Categorical Features Across Clusters\n{title}", y=1.02, fontsize=16)
            plt.show()
        else:
            print("No categorical features to visualize.")

        print("\n--- Cluster Sizes ---")
        cluster_counts = X_df['cluster_str'].value_counts().sort_index()
        print(cluster_counts)