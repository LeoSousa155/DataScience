from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from .BaseModel import BaseModel


class RandomForest(BaseModel):
    """
    A wrapper class for scikit-learn's RandomForest algorithms
    for both regression and classification tasks, implementing Bagging of Decision Trees.
    Inherits from BaseModel.
    """
    def __init__(self, problem_type='classification', **kwargs):
        """
        Initializes the RandomForest class.

        Args:
            problem_type (str, optional): The type of problem ('classification' or 'regression').
                                         Defaults to 'classification'.
            **kwargs: Additional arguments to be passed to the scikit-learn
                      RandomForest Classifier or Regressor. Common arguments include:
                      n_estimators (number of trees), criterion, max_depth,
                      min_samples_split, min_samples_leaf, max_features,
                      bootstrap (True for bagging), oob_score, n_jobs, random_state, etc.
        """
        self.problem_type = problem_type.lower()
        self.model = self._create_model(**kwargs)

    def _create_model(self, **kwargs):
        """
        Creates the appropriate RandomForest model instance
        based on the problem type.
        """
        if self.problem_type == 'classification':
            return RandomForestClassifier(**kwargs)
        elif self.problem_type == 'regression':
            return RandomForestRegressor(**kwargs)
        else:
            raise ValueError("problem_type must be 'classification' or 'regression'")

    def fit(self, X, y):
        """
        Trains the RandomForest model with the provided training data.

        Args:
            X (array-like): Training data (features).
            y (array-like): Training labels or target values.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """
        Performs predictions on the given data.

        Args:
            X (array-like): Data for prediction (features).

        Returns:
            array-like: Model predictions.
        """
        return self.model.predict(X)

    def predict_proba(self, X):
        """
        Predicts class probabilities for classification tasks.

        Args:
            X (array-like): Data for prediction (features).

        Returns:
            array-like: Predicted class probabilities.

        Raises:
            NotImplementedError: If the problem type is not 'classification'.
        """
        if self.problem_type != 'classification':
            raise NotImplementedError("predict_proba is only available for classification tasks.")
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the performance of the model on the provided test data.

        Args:
            X_test (array-like): Test data (features).
            y_test (array-like): Test labels or target values.

        Returns:
            dict: A dictionary containing the appropriate evaluation metrics
                  for the problem type.
        """
        y_pred = self.predict(X_test)
        results = {}
        if self.problem_type == 'regression':
            results['mean_squared_error'] = mean_squared_error(y_test, y_pred)
            results['r2_score'] = r2_score(y_test, y_pred)
            # You could add more regression metrics here if needed, e.g., MAE, RMSE
            # from sklearn.metrics import mean_absolute_error
            # results['mean_absolute_error'] = mean_absolute_error(y_test, y_pred)
            # results['root_mean_squared_error'] = mean_squared_error(y_test, y_pred, squared=False) # For RMSE
        elif self.problem_type == 'classification':
            results['accuracy'] = accuracy_score(y_test, y_pred)
            results['classification_report'] = classification_report(y_test, y_pred, output_dict=True) # output_dict=True makes it easier to parse
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist() # Convert to list for easier handling if serializing
            # You could add more classification metrics here if needed, e.g., precision, recall, f1-score for specific classes or averaged
            # from sklearn.metrics import precision_score, recall_score, f1_score
            # results['precision_macro'] = precision_score(y_test, y_pred, average='macro')
            # results['recall_macro'] = recall_score(y_test, y_pred, average='macro')
            # results['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        return results

    def get_feature_importance(self):
        """
        Returns the feature importances.

        Returns:
            numpy.ndarray: An array containing the importance of each feature.
                           Returns None if the model does not support feature importance (though RandomForest does).
        """
        # RandomForest models have feature_importances_ attribute
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None
