import numpy as np
from sklearn.preprocessing import StandardScaler
import pandas as pd
import config


class DataPreprocessor:
    @staticmethod
    def check_missing_values(data):
        """Check for missing values in the dataset."""
        return data.isnull().sum()

    @staticmethod
    def drop_missing_values(data):
        """Drop rows with missing values."""
        return data.dropna()

    @staticmethod
    def outlier_thresholds(dataframe, variable):
        """Determine the outlier thresholds for a given variable."""
        quartile1 = dataframe[variable].quantile(config.Config.OUTLIERS_LOWER_LIMIT)
        quartile3 = dataframe[variable].quantile(config.Config.OUTLIERS_UPPER_LIMIT)
        interquantile_range = quartile3 - quartile1
        up_limit = quartile3 + 1.5 * interquantile_range
        low_limit = quartile1 - 1.5 * interquantile_range
        return low_limit, up_limit

    @staticmethod
    def check_outlier(dataframe, col_name):
        """Check if there are outliers in the given column."""
        low_limit, up_limit = DataPreprocessor.outlier_thresholds(dataframe, col_name)
        return dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None)

    @staticmethod
    def replace_with_thresholds(dataframe, col_name):
        """Replace outliers with the thresholds."""
        low_limit, up_limit = DataPreprocessor.outlier_thresholds(dataframe, col_name)
        dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
        dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

    @staticmethod
    def preprocess_data(data):
        """Perform data preprocessing including outlier handling, feature engineering, and scaling."""
        # Separate categorical and numerical columns
        categorical_data = data.select_dtypes(include=['object'])
        numerical_data = data.select_dtypes(exclude=['object'])

        # Check and handle outliers
        for col in numerical_data.columns:
            if DataPreprocessor.check_outlier(data, col):
                DataPreprocessor.replace_with_thresholds(data, col)

        # Feature Engineering
        data['NEW_Hits'] = data['Hits'] / data['CHits'] + data['Hits']
        data['NEW_RBI'] = data['RBI'] / data['CRBI']
        data['NEW_Walks'] = data['Walks'] / data['CWalks']
        data['NEW_PutOuts'] = data['PutOuts'] * data['Years']
        data["Hits_Success"] = (data["Hits"] / data["AtBat"]) * 100
        data["NEW_CRBI*CATBAT"] = data['CRBI'] * data['CAtBat']
        data["NEW_RBI"] = data["RBI"] / data["CRBI"]
        data["NEW_Chits"] = data["CHits"] / data["Years"]
        data["NEW_CHmRun"] = data["CHmRun"] * data["Years"]
        data["NEW_CRuns"] = data["CRuns"] / data["Years"]
        data["NEW_Chits"] = data["CHits"] * data["Years"]
        data["NEW_RW"] = data["RBI"] * data["Walks"]
        data["NEW_RBWALK"] = data["RBI"] / data["Walks"]
        data["NEW_CH_CB"] = data["CHits"] / data["CAtBat"]
        data["NEW_CHm_CAT"] = data["CHmRun"] / data["CAtBat"]
        data['NEW_Diff_Atbat'] = data['AtBat'] - (data['CAtBat'] / data['Years'])
        data['NEW_Diff_Hits'] = data['Hits'] - (data['CHits'] / data['Years'])
        data['NEW_Diff_HmRun'] = data['HmRun'] - (data['CHmRun'] / data['Years'])
        data['NEW_Diff_Runs'] = data['Runs'] - (data['CRuns'] / data['Years'])
        data['NEW_Diff_RBI'] = data['RBI'] - (data['CRBI'] / data['Years'])
        data['NEW_Diff_Walks'] = data['Walks'] - (data['CWalks'] / data['Years'])

        # One-Hot Encoding
        data = pd.get_dummies(data, columns=categorical_data.columns, drop_first=True)

        # Scale the data
        scaler = StandardScaler()
        numerical_data = data.select_dtypes(exclude=['object'])
        data[numerical_data.columns] = scaler.fit_transform(data[numerical_data.columns])


        return data
