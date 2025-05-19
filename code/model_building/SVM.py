from sklearn.svm import SVR, SVC
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd
from typing import Dict


class SupportVectorMachine:
    """
    A wrapper class for scikit-learn's Support Vector Machine algorithms
    for both regression and classification tasks.
    """
    def __init__(self, problem_type='classification', grid_search=True, **kwargs):
        """
        Initializes the SupportVectorMachine class.

        Args:
            problem_type (str, optional): The type of problem ('classification' or 'regression').
                                         Defaults to 'classification'.
            grid_search (bool, optional): Whether to perform GridSearchCV for hyperparameter tuning.
                                          Defaults to True.
            **kwargs: Additional arguments to be passed to the scikit-learn
                      SVR or SVC, or GridSearchCV.  You can pass parameters like
                      'C', 'kernel', 'gamma', 'epsilon' (for SVR),
                      and 'cv', 'scoring' (for GridSearchCV).
        """
        self.problem_type = problem_type.lower()
        self.grid_search = grid_search
        self.model = self._create_model(**kwargs)
        self.best_model = None  # To store the best model after fitting
        self.cv_results = None #to store the results of the grid search

    def _create_model(self, **kwargs):
        """
        Creates the appropriate SVM model instance based on the problem type.
        """
        model_params = {}
        grid_params = {}
        for key, value in kwargs.items():
            if key in ['C', 'kernel', 'gamma', 'epsilon', 'class_weight']:
                model_params[key] = value
            elif key in ['cv', 'scoring', 'n_jobs']:
                grid_params[key] = value

        if self.problem_type == 'classification':
            if 'class_weight' not in model_params:
                model_params['class_weight'] = 'balanced'  # Default for classification
            model = SVC(**model_params)
        elif self.problem_type == 'regression':
            model = SVR(**model_params)
        else:
            raise ValueError("problem_type must be 'classification' or 'regression'")

        if self.grid_search:
            # Set default cv and scoring if not provided
            cv = grid_params.get('cv', 5)
            scoring = grid_params.get('scoring', 'accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error')
            n_jobs = grid_params.get('n_jobs', -1)

            param_grid = self._get_param_grid()  # Get parameter grid

            # Use a Pipeline to scale the data before applying SVM
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', model)  # Use the SVM model created above
            ])
            grid_search = GridSearchCV(pipeline, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)
            return grid_search
        else:
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('svm', model)
            ])
            return pipeline # Return the pipeline

    def _get_param_grid(self) -> Dict:
        """
        Define the parameter grid for GridSearchCV.  This method allows for
        more flexible parameter grids based on problem type.
        """
        if self.problem_type == 'classification':
            return {
                'svm__C': [0.1, 1, 10],
                'svm__kernel': ['linear', 'rbf'],
                'svm__gamma': ['scale', 'auto']
            }
        elif self.problem_type == 'regression':
            return {
                'svm__C': [0.1, 1, 10],
                'svm__kernel': ['linear', 'rbf'],
                'svm__epsilon': [0.01, 0.1, 0.2]
            }
        else:
            return {}

    def fit(self, X, y):
        """
        Trains the SVM model with the provided training data.

        Args:
            X (pd.DataFrame): Training data (features).
            y (pd.Series): Training labels or target values.
        """
        if self.grid_search:
            self.model.fit(X, y)
            self.best_model = self.model.best_estimator_ # Store the best model
            self.cv_results = self.model.cv_results_
        else:
            self.model.fit(X, y)
            self.best_model = self.model # Store the pipeline

    def predict(self, X):
        """
        Performs predictions on the given data.

        Args:
            X (pd.DataFrame): Data for prediction (features).

        Returns:
            np.ndarray: Model predictions.
        """
        if self.grid_search:
            return self.best_model.predict(X)
        else:
            return self.model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluates the performance of the model on the provided test data.

        Args:
            X_test (pd.DataFrame): Test data (features).
            y_test (pd.Series): Test labels or target values.

        Returns:
            dict: A dictionary containing the evaluation metrics.
        """
        y_pred = self.predict(X_test)
        results = {}
        if self.problem_type == 'regression':
            results['mean_squared_error'] = mean_squared_error(y_test, y_pred)
            results['r2_score'] = r2_score(y_test, y_pred)
            results['rmse'] = np.sqrt(results['mean_squared_error'])
        elif self.problem_type == 'classification':
            results['accuracy'] = accuracy_score(y_test, y_pred)
            results['classification_report'] = classification_report(y_test, y_pred)
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        return results

    def get_feature_importance(self):
        """
        Returns the feature importances (only for linear SVM).

        Returns:
            np.ndarray: An array containing the importance of each feature.
                           Returns None if the model does not support feature importance.
        """
        if self.best_model and hasattr(self.best_model.named_steps['svm'], 'coef_') and self.best_model.named_steps['svm'].kernel == 'linear':
            return self.best_model.named_steps['svm'].coef_[0]  # Access through the pipeline
        else:
            return None
