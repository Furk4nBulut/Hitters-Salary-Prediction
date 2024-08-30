# feature_selection/feature_selector.py
import config

class FeatureSelector:
    @staticmethod
    def select_features(data, target_column=config.Config.TARGET_COLUMN):
        """Select features and target variable."""
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        return X, y
