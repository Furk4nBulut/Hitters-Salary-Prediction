from config import Config
from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from feature_selector import FeatureSelector
from model_trainer import ModelTrainer
from visualization import Visualization
from sklearn.model_selection import train_test_split
import config
import pandas as pd

import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)

def main():
    # Set the configuration
    config.Config.set_config(
        dataset_path='data/hitters.csv',
        outliers_lower_limit=0.25,
        outliers_upper_limit=0.75,
        test_size=0.3,
        random_state=46,
        target_column='Salary'
    )

    # Load and preprocess data
    data = DataLoader.load_data(config.Config.DATASET_PATH)
    data = DataPreprocessor.preprocess_data(data)
    data = DataPreprocessor.drop_missing_values(data)

    # Select features and target variable
    X, y = FeatureSelector.select_features(data)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=config.Config.TEST_SIZE, random_state=config.Config.RANDOM_STATE)

    # Train and evaluate models
    initial_results = ModelTrainer.train_and_evaluate_all_models(X_train, y_train, X_test, y_test)
    tuned_results = ModelTrainer.tune_and_evaluate_models(X_train, y_train, X_test, y_test)

    # Export results to CSV
    ModelTrainer.export_results_to_csv(initial_results, tuned_results)

    # Print output path for debugging
    print(f"Output path: {config.Config.OUTPUT_PATH}")

    # Load results for visualization
    try:
        initial_df = pd.read_csv(config.Config.OUTPUT_PATH)
        tuned_df = pd.read_csv(config.Config.OUTPUT_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # Combine results
    combined_results = pd.merge(pd.DataFrame(initial_results), pd.DataFrame(tuned_results),
                                on='Model', suffixes=('_Initial', '_Tuned'))

    # Visualize results
    Visualization.plot_metrics(combined_results, save_as=Config.PLOT_METRICS)
    Visualization.plot_metric_comparison(pd.DataFrame(initial_results), pd.DataFrame(tuned_results), 'MSE',
                                         save_as=Config.PLOT_METRIC_COMPARISON)
    Visualization.plot_model_comparison(combined_results, save_as=Config.PLOT_MODEL_COMPARISON)


if __name__ == "__main__":
    main()
