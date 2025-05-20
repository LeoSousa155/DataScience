from sklearn.ensemble import AdaBoostClassifier, AdaBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from .BaseModel import BaseModel


class AdaBoost(BaseModel):
    """
    A wrapper class for scikit-learn's AdaBoost algorithms
    for both regression and classification tasks.
    AdaBoost (Adaptive Boosting) is a boosting meta-estimator that begins by fitting
    a model on the original dataset and then fits additional copies of the model
    on the same dataset but where the weights of incorrectly classified data points
    are adjusted such that subsequent models focus more on difficult cases.
    Inherits from BaseModel.
    """
    def __init__(self, problem_type='classification', base_estimator=None, **kwargs):
        """
        Initializes the AdaBoost class.

        Args:
            problem_type (str, optional): The type of problem ('classification' or 'regression').
                                         Defaults to 'classification'.
            base_estimator (estimator, optional): The base estimator from which the boosted
                                                  ensemble is built. If None, the default base
                                                  estimator is DecisionTreeClassifier(max_depth=1)
                                                  for classification and DecisionTreeRegressor(max_depth=3)
                                                  for regression. Defaults to None.
            **kwargs: Additional arguments to be passed to the scikit-learn
                      AdaBoost Classifier or Regressor. Common arguments include:
                      n_estimators (number of boosting stages), learning_rate, random_state, etc.
        """
        self.problem_type = problem_type.lower()
        self.base_estimator = base_estimator # Store the base estimator if provided
        self.model = self._create_model(**kwargs)

    def _create_model(self, **kwargs):
        """
        Creates the appropriate AdaBoost model instance
        based on the problem type and optional base estimator.
        """
        if self.problem_type == 'classification':
            # Default base estimator for AdaBoostClassifier is DecisionTreeClassifier(max_depth=1)
            # We pass the provided base_estimator or let scikit-learn use its default if None
            return AdaBoostClassifier(estimator=self.base_estimator, **kwargs)
        elif self.problem_type == 'regression':
            # Default base estimator for AdaBoostRegressor is DecisionTreeRegressor(max_depth=3)
            # We pass the provided base_estimator or let scikit-learn use its default if None
            return AdaBoostRegressor(estimator=self.base_estimator, **kwargs)
        else:
            raise ValueError("problem_type must be 'classification' or 'regression'")

    def fit(self, X, y):
        """
        Trains the AdaBoost model with the provided training data.

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
        Note: predict_proba for AdaBoostClassifier can be less reliable than for
        other models like RandomForest, as it's based on combining confidence scores
        of weak learners.

        Args:
            X (array-like): Data for prediction (features).

        Returns:
            array-like: Predicted class probabilities.

        Raises:
            NotImplementedError: If the problem type is not 'classification'.
        """
        if self.problem_type != 'classification':
            raise NotImplementedError("predict_proba is only available for classification tasks.")
        # Check if the underlying model supports predict_proba (AdaBoostClassifier does)
        if hasattr(self.model, 'predict_proba'):
             return self.model.predict_proba(X)
        else:
             raise NotImplementedError("The wrapped AdaBoost model does not support predict_proba.")


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
            # Use output_dict=True for easier parsing of classification report
            results['classification_report'] = classification_report(y_test, y_pred, output_dict=True)
            # Convert confusion matrix to list for potentially easier handling if serializing
            results['confusion_matrix'] = confusion_matrix(y_test, y_pred).tolist()
            # You could add more classification metrics here if needed
            # from sklearn.metrics import precision_score, recall_score, f1_score
            # results['precision_macro'] = precision_score(y_test, y_pred, average='macro')
            # results['recall_macro'] = recall_score(y_test, y_pred, average='macro')
            # results['f1_macro'] = f1_score(y_test, y_pred, average='macro')
        return results

    def get_feature_importance(self):
        """
        Returns the feature importances.
        Note: Feature importances for AdaBoost are calculated based on the
        contributions of the base estimators.

        Returns:
            numpy.ndarray: An array containing the importance of each feature.
                           Returns None if the model does not support feature importance.
        """
        # AdaBoost models have feature_importances_ attribute if the base estimator does
        # and the model is a classifier or regressor.
        if hasattr(self.model, 'feature_importances_'):
            return self.model.feature_importances_
        else:
             # This might happen if the base estimator doesn't support feature_importances_
             return None