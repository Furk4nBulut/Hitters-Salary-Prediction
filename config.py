import os

class Config:
    # Default configuration values
    DATASET_PATH = 'data/hitters.csv'
    OUTPUT_PATH = 'output/result.csv'
    OUTLIERS_LOWER_LIMIT = 0.35
    OUTLIERS_UPPER_LIMIT = 0.75
    TEST_SIZE = 0.35
    RANDOM_STATE = 46
    TARGET_COLUMN = 'Salary'
    METRICS = ['MSE', 'RMSE', 'MAE', 'R2 Score']

    # plot_metrics
    PLOT_METRICS = 'output/metrics_plot.png'
    PLOT_METRIC_COMPARISON = 'output/mse_comparison_plot.png'
    PLOT_MODEL_COMPARISON = 'output/model_comparison_plot.png'

    # Model-specific default hyperparameters
    HYPERPARAMETERS = {
        "LinearRegression": {},  # No hyperparameters for Linear Regression
        "RidgeRegression": {
            'alpha': [0.1, 1.0, 10.0]
        },
        "LassoRegression": {
            'alpha': [0.1, 1.0, 10.0]
        },
        "ElasticNet": {
            'alpha': [0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9]
        },
        "PolynomialRegression": {
            'polynomialfeatures__degree': [2, 3, 4]  # Degree of polynomial features
        },
        "DecisionTree": {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        "RandomForestRegressor": {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        },
        "GradientBoostingRegressor": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 5, 7]
        },
        "SupportVectorMachine": {
            'C': [0.1, 1.0, 10.0],
            'epsilon': [0.1, 0.2, 0.3],
            'kernel': ['linear', 'rbf']
        },
        "KNearestNeighbors": {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # p=1 for Manhattan distance, p=2 for Euclidean distance
        },
        "XGBoost": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 6, 10]
        },
        "LightGBM": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'num_leaves': [31, 63, 127]
        },
        "CatBoost": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.2],
            'depth': [6, 8, 10]
        }
    }

    @classmethod
    def set_config(cls, dataset_path=None, outliers_lower_limit=None, outliers_upper_limit=None, test_size=None, random_state=None, target_column=None, output_path=None, plot_metrics=None, plot_metric_comparison=None, plot_model_comparison=None):
        """Update configuration values."""
        if dataset_path is not None:
            cls.DATASET_PATH = dataset_path
        if outliers_lower_limit is not None:
            cls.OUTLIERS_LOWER_LIMIT = outliers_lower_limit
        if outliers_upper_limit is not None:
            cls.OUTLIERS_UPPER_LIMIT = outliers_upper_limit
        if test_size is not None:
            cls.TEST_SIZE = test_size
        if random_state is not None:
            cls.RANDOM_STATE = random_state
        if target_column is not None:
            cls.TARGET_COLUMN = target_column
        if output_path is not None:
            cls.OUTPUT_PATH = output_path
        if plot_metrics is not None:
            cls.PLOT_METRICS = plot_metrics
        if plot_metric_comparison is not None:
            cls.PLOT_METRIC_COMPARISON = plot_metric_comparison
        if plot_model_comparison is not None:
            cls.PLOT_MODEL_COMPARISON = plot_model_comparison
