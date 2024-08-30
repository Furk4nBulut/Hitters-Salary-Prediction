# main.py

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from feature_selector import FeatureSelector
from model_trainer import ModelTrainer
from sklearn.model_selection import train_test_split


def main():
    # File path
    file_path = 'data/hitters.csv'

    # Load and preprocess data
    data = DataLoader.load_data(file_path)
    data = DataPreprocessor.drop_missing_values(data)
    data = DataPreprocessor.preprocess_data(data)

    # Select features and target variable
    X, y = FeatureSelector.select_features(data)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=46)

    # Train and evaluate the model
    model = ModelTrainer.train_model(X_train, y_train)
    ModelTrainer.evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
