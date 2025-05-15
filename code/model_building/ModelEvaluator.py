import numpy as np
from sklearn import metrics
from sklearn.model_selection import KFold, StratifiedKFold
from typing import Optional, List, Union
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
        cv_method: The cross-validation method.  Can be one of 'kfold',
            'stratifiedkfold', or None.  If None, no cross-validation is performed.
        cv_folds: The number of folds to use for cross-validation.  Ignored if
            `cv_method` is None.  Defaults to 5.
        random_state:  The random state to use for any random operations,
            including cross-validation splitting.  Defaults to 42.
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
        cv_method: Optional[str] = 'kfold',
        cv_folds: int = 5,
        random_state: int = 42,
        pos_label: int = 1,
    ):
        self.model = model
        self.task_type = task_type
        self.metrics = metrics if metrics else (
            ['accuracy'] if task_type in ['binary', 'multiclass'] else ['r2']
        )
        self.cv_method = cv_method
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.pos_label = pos_label

        self._validate_task_type()
        self._validate_metrics()
        self._validate_cv_method()

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

    def _validate_cv_method(self):
        """Validates that the cross-validation method is one of the supported types."""
        if self.cv_method not in ['kfold', 'stratifiedkfold', None]:
            raise ValueError(
                f"Invalid cv_method: {self.cv_method}.  Must be one of "
                "'kfold', 'stratifiedkfold', or None."
            )

    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: Optional[Union[int, KFold, StratifiedKFold]] = None,
        return_predictions: bool = False
    ) -> dict:
        """
        Evaluates the model on the given data, optionally using cross-validation.

        Args:
            X: The input features as a pandas DataFrame.
            y: The target values as a pandas Series.
            cv:  Cross-validation strategy.  If None, uses the cv_method and
                 cv_folds specified in the constructor.  If not None, overrides
                 the constructor settings. Can be an integer (number of folds),
                 a KFold object, or a StratifiedKFold object.
            return_predictions: Whether to return the predictions and (if applicable)
                predicted probabilities along with the metrics. Defaults to False.

        Returns:
            A dictionary containing the calculated metrics.  If cross-validation
            is used, the dictionary contains the mean and standard deviation of
            each metric.  If cross-validation is not used, the dictionary
            contains the metric values on the single provided dataset.
            If return_predictions is True, the dictionary also includes
            'predictions' and (for classification) 'probabilities' keys.
        """

        if cv is None:
            cv = self._get_cv_strategy()

        if cv:
            return self._evaluate_with_cv(X, y, cv, return_predictions)
        else:
            return self._evaluate_without_cv(X, y, return_predictions)

    def _get_cv_strategy(self) -> Optional[Union[KFold, StratifiedKFold, int]]:
        """
        Determines the cross-validation strategy based on the object's attributes.

        Returns:
            An appropriate cross-validation object (KFold or StratifiedKFold)
            or None if no cross-validation is to be performed.  Returns an int
            if self.cv_method is specified, and the user did not pass a cv
            argument to evaluate().
        """
        if self.cv_method is None:
            return None

        if self.cv_method == 'kfold':
            return KFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
        elif self.cv_method == 'stratifiedkfold':
            return StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
        return None  # Should never reach here, but included for completeness

    def _evaluate_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: Union[int, KFold, StratifiedKFold],
        return_predictions: bool
    ) -> dict:
        """
        Evaluates the model using cross-validation.

        Args:
            X: The input features.
            y: The target values.
            cv: The cross-validation strategy (instance of KFold or StratifiedKFold).
            return_predictions: Whether to return predictions.

        Returns:
            A dictionary containing the mean and standard deviation of the
            calculated metrics across the cross-validation folds.
            Optionally includes predictions from each fold.
        """
        results = {metric: [] for metric in self.metrics}
        predictions = []
        probabilities = []

        # Use the passed cv object, whether it's an int, or a KFold/StratifiedKFold
        for fold_idx, (train_index, test_index) in enumerate(cv.split(X, y)):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            if return_predictions:
                predictions.append(y_pred)

            if self.task_type in ['binary', 'multiclass']:
                try:
                    y_prob = self.model.predict_proba(X_test)
                    if return_predictions:
                        probabilities.append(y_prob)
                except AttributeError:
                    y_prob = None
                    if 'log_loss' in self.metrics:
                        print(
                            "Warning: Model does not have predict_proba method. "
                            "Log loss cannot be calculated."
                        )
            else:
                y_prob = None

            fold_results = self._calculate_metrics(y_test, y_pred, y_prob)
            for metric, value in fold_results.items():
                results[metric].append(value)

        # Calculate mean and standard deviation for each metric
        mean_results = {
            f'{metric}_mean': np.mean(values)
            for metric, values in results.items()
        }
        std_results = {
            f'{metric}_std': np.std(values)
            for metric, values in results.items()
        }

        final_results = {**mean_results, **std_results}  # Merge the two dicts

        if return_predictions:
            final_results['predictions'] = predictions
            if self.task_type in ['binary', 'multiclass']:
                final_results['probabilities'] = probabilities
        return final_results

    def _evaluate_without_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        return_predictions: bool
    ) -> dict:
        """
        Evaluates the model on the given data without cross-validation.

        Args:
            X: The input features.
            y: The target values.
            return_predictions: Whether to return predictions.

        Returns:
            A dictionary containing the calculated metrics.
            Optionally includes the predictions.
        """
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
                        y_true, y_pred, average=average, pos_label=self.pos_label
                    )
                elif metric == 'recall':
                    metrics_dict[metric] = metrics.recall_score(
                        y_true, y_pred, average=average, pos_label=self.pos_label
                    )
                elif metric == 'f1':
                    metrics_dict[metric] = metrics.f1_score(
                        y_true, y_pred, average=average, pos_label=self.pos_label
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
            elif metric == 'log_loss':
                metrics_dict[metric] = metrics.log_loss(y_true, y_prob)
            elif metric == 'brier_score':
                metrics_dict[metric] = metrics.brier_score_loss(y_true, y_prob[:, 1])
            elif metric in ['precision_macro', 'precision_micro', 'recall_macro',
                             'recall_micro', 'f1_macro', 'f1_micro']:
                average = metric.split('_')[-1]  # Extract 'macro' or 'micro'
                if 'precision' in metric:
                    metrics_dict[metric] = metrics.precision_score(y_true, y_pred, average=average)
                elif 'recall' in metric:
                    metrics_dict[metric] = metrics.recall_score(y_true, y_pred, average=average)
                elif 'f1' in metric:
                    metrics_dict[metric] = metrics.f1_score(y_true, y_pred, average=average)
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
