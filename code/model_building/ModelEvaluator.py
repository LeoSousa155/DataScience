import numpy as np
from sklearn import metrics
from typing import Optional, List
import pandas as pd

class ModelEvaluator:
    """
    A class for evaluating machine learning models.  It supports binary, multiclass,
    and regression tasks, and provides a variety of evaluation metrics,
    cross-validation, and visualization options.

    Attributes:
        model: The trained machine learning model to evaluate.  Can be any object
            that implements a `predict` method for classification or regression,
            and optionally a `predict_proba` method for classification.
        task_type:  The type of machine learning task.  Must be one of
            'binary', 'multiclass', or 'regression'.
        metrics: A list of metric names (strings) to calculate.
            See the `supported_metrics` property for available metrics.
            Defaults to ['accuracy'] for classification and ['r2'] for regression.        
        random_state:  The random state to use for any random operations.
            Defaults to 42.
        pos_label:  The positive class label for binary classification.  Used for
            metrics like precision, recall, and F1-score.  Defaults to 1.
    """
    supported_metrics = {
        'binary': [
            'accuracy', 'precision', 'recall', 'f1', 'auc', 'average_precision',
            'log_loss', 'brier_score'
        ],
        'multiclass': [
            'accuracy', 'precision_macro', 'precision_micro', 'recall_macro',
            'recall_micro', 'f1_macro', 'f1_micro', 'log_loss'
        ],
        'regression': [
            'r2', 'mse', 'rmse', 'mae', 'explained_variance', 'max_error'
        ]
    }

    def __init__(
        self,
        model: object,
        task_type: str,
        metrics: Optional[List[str]] = None,
        random_state: int = 42,
        pos_label: int = 1,
    ):
        self.model = model
        self.task_type = task_type
        self.metrics = metrics if metrics else (
            ['accuracy'] if task_type in ['binary', 'multiclass'] else ['r2']
        )
        self.random_state = random_state
        self.pos_label = pos_label

        self._validate_task_type()
        self._validate_metrics()

    def _validate_task_type(self):
        """Validates that the task type is one of the supported types."""
        if self.task_type not in ['binary', 'multiclass', 'regression']:
            raise ValueError(
                f"Invalid task type: {self.task_type}.  Must be one of "
                "'binary', 'multiclass', or 'regression'."
            )

    def _validate_metrics(self):
        """Validates that the specified metrics are supported for the task type."""
        for metric in self.metrics:
            if metric not in self.supported_metrics[self.task_type]:
                raise ValueError(
                    f"Invalid metric: {metric} for task type: {self.task_type}.  "
                    f"Supported metrics are: {self.supported_metrics[self.task_type]}"
                )

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        return_predictions: bool = False
    ) -> dict:
        """
        Evaluates the model on the given data.

        Args:
            X: The input features as a pandas DataFrame.
            y: The target values as a pandas Series.
            return_predictions: Whether to return the predictions and (if applicable)
                predicted probabilities along with the metrics. Defaults to False.

        Returns:
            A dictionary containing the calculated metrics on the provided dataset.
            If return_predictions is True, the dictionary also includes
            'predictions' and (for classification) 'probabilities' keys.
        """
        return self._evaluate_without_cv(X, y, return_predictions)

    def _evaluate_without_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        return_predictions: bool
    ) -> dict:
        """
        Evaluates the model on the given data.

        Args:
            X: The input features.
            y: The target values.
            return_predictions: Whether to return predictions.

        Returns:
            A dictionary containing the calculated metrics.
            Optionally includes the predictions.
        """
        # Assuming the model is already trained if no CV is used by the evaluator itself.
        # If the model needs training, it should be done before calling evaluate.
        y_pred = self.model.predict(X)

        if self.task_type in ['binary', 'multiclass']:
            try:
                y_prob = self.model.predict_proba(X)
            except AttributeError:
                y_prob = None
                if 'log_loss' in self.metrics:
                    print(
                        "Warning: Model does not have predict_proba method. "
                        "Log loss cannot be calculated."
                    )
        else:
            y_prob = None

        results = self._calculate_metrics(y, y_pred, y_prob)
        if return_predictions:
            results['predictions'] = y_pred
            if self.task_type in ['binary', 'multiclass']:
                results['probabilities'] = y_prob
        return results

    def _calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None
    ) -> dict:
        """
        Calculates the specified evaluation metrics.

        Args:
            y_true: The true target values.
            y_pred: The predicted target values.
            y_prob: The predicted probabilities (optional, required for some metrics).

        Returns:
            A dictionary containing the calculated metrics.
        """
        metrics_dict = {}
        for metric in self.metrics:
            if metric == 'accuracy':
                metrics_dict[metric] = metrics.accuracy_score(y_true, y_pred)
            elif metric in ['precision', 'recall', 'f1']:
                average = 'binary' if self.task_type == 'binary' else 'macro'
                if metric == 'precision':
                    metrics_dict[metric] = metrics.precision_score(
                        y_true, y_pred, average=average, pos_label=self.pos_label, zero_division=0
                    )
                elif metric == 'recall':
                    metrics_dict[metric] = metrics.recall_score(
                        y_true, y_pred, average=average, pos_label=self.pos_label, zero_division=0
                    )
                elif metric == 'f1':
                    metrics_dict[metric] = metrics.f1_score(
                        y_true, y_pred, average=average, pos_label=self.pos_label, zero_division=0
                    )
            elif metric == 'auc':
                if self.task_type == 'binary':
                    metrics_dict[metric] = metrics.roc_auc_score(y_true, y_prob[:, 1])
                else:
                    # Handle multiclass case.
                    metrics_dict[metric] = metrics.roc_auc_score(y_true, y_prob, multi_class='ovr')
            elif metric == 'average_precision':
                metrics_dict[metric] = metrics.average_precision_score(
                    y_true, y_prob[:, 1]
                )
            elif metric == 'log_loss' and y_prob is not None:
                metrics_dict[metric] = metrics.log_loss(y_true, y_prob)
            elif metric == 'brier_score':
                metrics_dict[metric] = metrics.brier_score_loss(y_true, y_prob[:, 1])
            elif metric in ['precision_macro', 'precision_micro', 'recall_macro',
                             'recall_micro', 'f1_macro', 'f1_micro']:
                average = metric.split('_')[-1]  # Extract 'macro' or 'micro'
                if 'precision' in metric:
                    metrics_dict[metric] = metrics.precision_score(y_true, y_pred, average=average, zero_division=0)
                elif 'recall' in metric:
                    metrics_dict[metric] = metrics.recall_score(y_true, y_pred, average=average, zero_division=0)
                elif 'f1' in metric:
                    metrics_dict[metric] = metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
            elif metric == 'r2':
                metrics_dict[metric] = metrics.r2_score(y_true, y_pred)
            elif metric == 'mse':
                metrics_dict[metric] = metrics.mean_squared_error(y_true, y_pred)
            elif metric == 'rmse':
                metrics_dict[metric] = np.sqrt(metrics.mean_squared_error(y_true, y_pred))
            elif metric == 'mae':
                metrics_dict[metric] = metrics.mean_absolute_error(y_true, y_pred)
            elif metric == 'explained_variance':
                metrics_dict[metric] = metrics.explained_variance_score(y_true, y_pred)
            elif metric == 'max_error':
                metrics_dict[metric] = metrics.max_error(y_true, y_pred)
        return metrics_dict
