from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import BaseModel


class DecisionTree(BaseModel):
    """
    A wrapper class for scikit-learn's Decision Tree algorithms
    for both regression and classification tasks.
    """
    def __init__(self, problem_type='classification', **kwargs):
        """
        Initializes the CustomDecisionTree class.

        Args:
            problem_type (str, optional): The type of problem ('classification' or 'regression').
                                         Defaults to 'classification'.
            **kwargs: Additional arguments to be passed to the scikit-learn
                      Decision Tree Classifier or Regressor.
        """
        self.problem_type = problem_type.lower()
        self.model = self._create_model(**kwargs)

    def _create_model(self, **kwargs):
        """
        Creates the appropriate Decision Tree model instance
        based on the problem type.
        """
        if self.problem_type == 'classification':
            return DecisionTreeClassifier(**kwargs)
        elif self.problem_type == 'regression':
            return DecisionTreeRegressor(**kwargs)
        else:
            raise ValueError("problem_type must be 'classification' or 'regression'")

    def fit(self, X, y):
        """
        Trains the Decision Tree model with the provided training data.

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
        elif self.problem_type == 'classification':
            results['accuracy'] = accuracy_score(y_test, y_pred)
            results['classification_report'] = classification_report(y_test, y_pred)
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        return results

    def get_feature_importance(self):
        """
        Returns the feature importances (only for tree-based models).

        Returns:
            numpy.ndarray: An array containing the importance of each feature.
                           Returns None if the model does not support feature importance.
        """
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
            return None