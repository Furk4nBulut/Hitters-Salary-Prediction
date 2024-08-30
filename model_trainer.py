import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import numpy as np
import config
from config import Config
from hyperparameter_tuner import HyperparameterTuner
from sklearn.ensemble import HistGradientBoostingRegressor


class ModelTrainer:

    @staticmethod
    def train_and_evaluate_all_models(X_train, y_train, X_test, y_test):
        """Train and evaluate all models."""
        results = []
        models = {
            "LinearRegression": LinearRegression(),
            "RidgeRegression": Ridge(),
            "LassoRegression": Lasso(),
            "ElasticNet": ElasticNet(),
            "PolynomialRegression": make_pipeline(PolynomialFeatures(), LinearRegression()),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "SupportVectorMachine": SVR(),
            "KNearestNeighbors": KNeighborsRegressor(),
            "XGBoost": XGBRegressor(),
            "LightGBM": LGBMRegressor(),
            "CatBoost": CatBoostRegressor()
        }

        for model_name, model in models.items():
            print(f"\nTraining and evaluating {model_name}...")
            # Train model
            model.fit(X_train, y_train)
            metrics = ModelTrainer.evaluate_model(model, X_test, y_test, return_metrics=True)
            metrics['Model'] = model_name
            metrics['Type'] = 'Initial'
            results.append(metrics)

        return results

    @staticmethod
    def tune_and_evaluate_models(X_train, y_train, X_test, y_test):
        """Tune hyperparameters and evaluate all models."""
        results = []
        models = {
            "LinearRegression": LinearRegression(),
            "RidgeRegression": Ridge(),
            "LassoRegression": Lasso(),
            "ElasticNet": ElasticNet(),
            "PolynomialRegression": make_pipeline(PolynomialFeatures(), LinearRegression()),
            "DecisionTree": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "SupportVectorMachine": SVR(),
            "KNearestNeighbors": KNeighborsRegressor(),
            "XGBoost": XGBRegressor(),
            "LightGBM": LGBMRegressor(),
            "CatBoost": CatBoostRegressor()
        }

        for model_name, model in models.items():
            param_grid = config.Config.HYPERPARAMETERS.get(model_name.replace(' ', ''))
            if param_grid:
                print(f"\nTuning hyperparameters for {model_name}...")
                best_model, best_params, best_score = HyperparameterTuner.tune_model(
                    model, param_grid, X_train, y_train
                )
                metrics = ModelTrainer.evaluate_model(best_model, X_test, y_test, return_metrics=True)
                metrics['Model'] = model_name
                metrics['Type'] = 'Tuned'
                results.append(metrics)
            else:
                print(f"No hyperparameters defined for {model_name}.")

        return results

    @staticmethod
    def evaluate_model(model, X_test, y_test, return_metrics=False):
        """Evaluate the model and return performance metrics."""
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2 Score': r2
        }

        if return_metrics:
            return metrics

        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAE: {mae}")
        print(f"R2 Score: {r2}")

    @staticmethod
    def export_results_to_csv(initial_results, tuned_results, filename=Config.OUTPUT_PATH):
        """Export results to a CSV file with initial and tuned results side by side."""
        # Initial results DataFrame
        initial_df = pd.DataFrame(initial_results)
        initial_df = initial_df.rename(columns=lambda x: f'Initial_{x}')

        # Tuned results DataFrame
        if tuned_results:
            tuned_df = pd.DataFrame(tuned_results)
            tuned_df = tuned_df.rename(columns=lambda x: f'Tuned_{x}')

            # Merge initial and tuned results
            combined_df = pd.concat([initial_df, tuned_df], axis=1)
        else:
            # If no tuned results, only export initial results
            combined_df = initial_df

        # Export to CSV
        combined_df.to_csv(filename, index=False)
        print(f"Results exported to {filename}")

    @staticmethod
    def linear_regression_model(X_train, y_train, X_test, y_test):
        """Train and evaluate the Linear Regression model."""
        model = LinearRegression()
        model.fit(X_train, y_train)
        ModelTrainer.evaluate_model(model, X_test, y_test)
        return model

    @staticmethod
    def ridge_model(X_train, y_train, X_test, y_test, alpha=1.0):
        """Train and evaluate the Ridge model."""
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        ModelTrainer.evaluate_model(model, X_test, y_test)
        return model

    @staticmethod
    def lasso_model(X_train, y_train, X_test, y_test, alpha=1.0):
        """Train and evaluate the Lasso model."""
        model = Lasso(alpha=alpha)
        model.fit(X_train, y_train)
        ModelTrainer.evaluate_model(model, X_test, y_test)
        return model

    @staticmethod
    def elastic_net_model(X_train, y_train, X_test, y_test, alpha=1.0, l1_ratio=0.5):
        """Train and evaluate the Elastic Net model."""
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
        model.fit(X_train, y_train)
        ModelTrainer.evaluate_model(model, X_test, y_test)
        return model

    @staticmethod
    def polynomial_regression_model(X_train, y_train, X_test, y_test, degree=2):
        """Train and evaluate the Polynomial Regression model."""
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(X_train, y_train)
        ModelTrainer.evaluate_model(model, X_test, y_test)
        return model

    @staticmethod
    def decision_tree_model(X_train, y_train, X_test, y_test, max_depth=None):
        """Train and evaluate the Decision Tree model."""
        model = DecisionTreeRegressor(max_depth=max_depth)
        model.fit(X_train, y_train)
        ModelTrainer.evaluate_model(model, X_test, y_test)
        return model

    @staticmethod
    def random_forest_model(X_train, y_train, X_test, y_test, n_estimators=100):
        """Train and evaluate the Random Forest model."""
        model = RandomForestRegressor(n_estimators=n_estimators)
        model.fit(X_train, y_train)
        ModelTrainer.evaluate_model(model, X_test, y_test)
        return model

    @staticmethod
    def gradient_boosting_model(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1,
                                early_stopping_rounds=10):
        """Train and evaluate the Gradient Boosting model with early stopping."""
        model = HistGradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
        model.fit(X_train, y_train, validation_data=(X_test, y_test), early_stopping_rounds=early_stopping_rounds)
        ModelTrainer.evaluate_model(model, X_test, y_test)
        return model

    @staticmethod
    def support_vector_machine_model(X_train, y_train, X_test, y_test, kernel='rbf'):
        """Train and evaluate the Support Vector Machine model."""
        model = SVR(kernel=kernel)
        model.fit(X_train, y_train)
        ModelTrainer.evaluate_model(model, X_test, y_test)
        return model

    @staticmethod
    def k_nearest_neighbors_model(X_train, y_train, X_test, y_test, n_neighbors=5):
        """Train and evaluate the K-Nearest Neighbors model."""
        model = KNeighborsRegressor(n_neighbors=n_neighbors)
        model.fit(X_train, y_train)
        ModelTrainer.evaluate_model(model, X_test, y_test)
        return model

    @staticmethod
    def xgboost_model(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, early_stopping_rounds=10):
        """Train and evaluate the XGBoost model with early stopping."""
        model = XGBRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
        evals = [(X_train, y_train), (X_test, y_test)]
        model.fit(X_train, y_train, eval_set=evals, early_stopping_rounds=early_stopping_rounds, verbose=False)
        ModelTrainer.evaluate_model(model, X_test, y_test)
        return model

    @staticmethod
    def lightgbm_model(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, early_stopping_rounds=10):
        """Train and evaluate the LightGBM model with early stopping."""
        model = LGBMRegressor(n_estimators=n_estimators, learning_rate=learning_rate)
        evals = [(X_train, y_train), (X_test, y_test)]
        model.fit(X_train, y_train, eval_set=evals, early_stopping_rounds=early_stopping_rounds, verbose=False)
        ModelTrainer.evaluate_model(model, X_test, y_test)
        return model

    @staticmethod
    def catboost_model(X_train, y_train, X_test, y_test, n_estimators=100, learning_rate=0.1, early_stopping_rounds=10):
        """Train and evaluate the CatBoost model with early stopping."""
        model = CatBoostRegressor(n_estimators=n_estimators, learning_rate=learning_rate, verbose=0)
        model.fit(X_train, y_train, eval_set=(X_test, y_test), early_stopping_rounds=early_stopping_rounds)
        ModelTrainer.evaluate_model(model, X_test, y_test)
        return model
