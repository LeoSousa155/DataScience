import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from typing import Dict, Union, Any

from .BaseModel import BaseModel

class LinearModels(BaseModel):
    """
    A wrapper class for scikit-learn's Linear Models
    for both regression and classification tasks.
    NOTE: This version requires manual preprocessing of data (X) before fitting or predicting.
    It does NOT handle numerical scaling or categorical encoding internally.
    """

    def __init__(self,
                 problem_type: str,
                 model_name: str,
                 grid_search: bool = True,
                 random_state: int = None,
                 custom_param_grid: Dict = None,
                 **kwargs):
        self.problem_type = problem_type.lower()
        self.model_name = model_name.lower()
        self.grid_search = grid_search
        self.random_state = random_state
        self.custom_param_grid = custom_param_grid

        self.model: Union[Any, GridSearchCV, None] = None
        self.best_model: Union[Any, None] = None
        self.cv_results: Union[Dict, None] = None

        self.model = self._create_model(**kwargs)


    def _create_model(self, **kwargs) -> Union[GridSearchCV, Any]:
        model_params = {}
        grid_params = {}

        essential_model_config_params = {
            'fit_intercept', 'positive', 'copy_X', 'n_jobs',
            'max_iter', 'tol', 'warm_start', 'precompute', 'selection',
            'solver', 'class_weight'
        }

        grid_search_config_params = {
            'cv', 'scoring', 'n_jobs', 'verbose', 'error_score', 'return_train_score'
        }

        if self.random_state is not None:
            if self.model_name in ['logistic', 'ridge', 'lasso', 'elasticnet']:
                if 'random_state' not in kwargs:
                    model_params['random_state'] = self.random_state

        for key, value in kwargs.items():
            if key in essential_model_config_params:
                model_params[key] = value
            elif key in grid_search_config_params:
                grid_params[key] = value
            else:
                model_params[key] = value

        base_model = None
        if self.problem_type == 'classification':
            if self.model_name == 'logistic':
                model_params.setdefault('class_weight', 'balanced')
                model_params.setdefault('max_iter', 1000)
                if self.grid_search:
                    model_params.setdefault('solver', 'saga')
                else:
                    model_params.setdefault('solver', 'lbfgs')
                base_model = LogisticRegression(**model_params)
            else:
                raise ValueError(f"Model '{self.model_name}' not supported for classification.")
        elif self.problem_type == 'regression':
            if self.model_name == 'linear':
                base_model = LinearRegression(**model_params)
            elif self.model_name == 'ridge':
                base_model = Ridge(**model_params)
            elif self.model_name == 'lasso':
                base_model = Lasso(**model_params)
            elif self.model_name == 'elasticnet':
                base_model = ElasticNet(**model_params)
            else:
                raise ValueError(f"Model '{self.model_name}' not supported for regression.")
        else:
            raise ValueError("problem_type must be 'classification' or 'regression'")

        if self.grid_search:
            cv = grid_params.get('cv', 5)
            scoring = grid_params.get('scoring', 'accuracy' if self.problem_type == 'classification' else 'neg_mean_squared_error')
            n_jobs = grid_params.get('n_jobs', -1)
            verbose = grid_params.get('verbose', 0)
            error_score = grid_params.get('error_score', np.nan)
            return_train_score = grid_params.get('return_train_score', True)

            param_grid = self.custom_param_grid if self.custom_param_grid is not None else self._get_param_grid()

            grid_search_model = GridSearchCV(base_model, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose, error_score=error_score, return_train_score=return_train_score)
            return grid_search_model
        else:
            return base_model

    def _get_param_grid(self) -> Dict:
        if self.problem_type == 'classification' and self.model_name == 'logistic':
            return {
                'C': [0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2', 'elasticnet', None],
                'solver': ['saga'],
                'l1_ratio': [0.1, 0.5, 0.9, None]
            }
        elif self.problem_type == 'regression':
            if self.model_name == 'linear':
                return {}
            elif self.model_name == 'ridge':
                return {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
                }
            elif self.model_name == 'lasso':
                return {
                    'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
                }
            elif self.model_name == 'elasticnet':
                return {
                    'alpha': [0.01, 0.1, 1.0, 10.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
        return {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.best_model = None
        self.cv_results = None

        if self.grid_search:
            print(f"Iniciando GridSearchCV para {self.model_name.upper()}...")
            try:
                self.model.fit(X, y)
                self.best_model = self.model.best_estimator_
                self.cv_results = self.model.cv_results_
                print(f"GridSearchCV finalizado para {self.model_name.upper()}. Melhores parâmetros: {self.model.best_params_}")
            except Exception as e:
                print(f"ERRO: GridSearchCV para {self.model_name.upper()} falhou durante o fit: {e}")
        else:
            print(f"Treinando modelo {self.model_name.upper()}...")
            try:
                self.model.fit(X, y)
                self.best_model = self.model
                print(f"Modelo {self.model_name.upper()} treinado.")
            except Exception as e:
                print(f"ERRO: Treinamento direto do modelo {self.model_name.upper()} falhou: {e}")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if self.best_model is None:
            raise RuntimeError("Modelo não foi treinado ainda. Chame .fit() primeiro.")
        return self.best_model.predict(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        if self.best_model is None:
            print("AVISO: Modelo não treinado ou falhou no treinamento. Não é possível avaliar.")
            return {}

        y_pred = self.predict(X_test)
        results = {}
        if self.problem_type == 'regression':
            results['mean_squared_error'] = mean_squared_error(y_test, y_pred)
            results['r2_score'] = r2_score(y_test, y_pred)
            results['rmse'] = np.sqrt(results['mean_squared_error'])
        elif self.problem_type == 'classification':
            results['accuracy'] = accuracy_score(y_test, y_pred)
            if len(np.unique(y_test)) > 1:
                results['classification_report'] = classification_report(y_test, y_pred, zero_division=0)
                results['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            else:
                print("AVISO: Apenas uma classe presente em y_test. classification_report e confusion_matrix não calculados.")
                results['classification_report'] = "N/A"
                results['confusion_matrix'] = "N/A"
        return results

    def get_feature_importance(self) -> Union[np.ndarray, None]:
        if self.best_model and hasattr(self.best_model, 'coef_'):
            return self.best_model.coef_
        else:
            print("Importância das features (coeficientes) não disponível para este modelo ou kernel.")
            return None