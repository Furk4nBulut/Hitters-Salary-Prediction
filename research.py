# İş Problemi

# Maaş bilgileri ve 1986 yılına ait kariyer istatistikleri paylaşılan beyzbol
# oyuncularının maaş tahminleri için bir makine öğrenmesi projesi gerçekleştirilebilir mi?

# Veri seti hikayesi

# Bu veri seti orijinal olarak Carnegie Mellon Üniversitesi'nde bulunan StatLib kütüphanesinden alınmıştır.
# Veri seti 1988 ASA Grafik Bölümü Poster Oturumu'nda kullanılan verilerin bir parçasıdır.
# Maaş verileri orijinal olarak Sports Illustrated, 20 Nisan 1987'den alınmıştır.
# 1986 ve kariyer istatistikleri, Collier Books, Macmillan Publishing Company, New York tarafından yayınlanan
# 1987 Beyzbol Ansiklopedisi Güncellemesinden elde edilmiştir.


# AtBat: 1986-1987 sezonunda bir beyzbol sopası ile topa yapılan vuruş sayısı
# Hits: 1986-1987 sezonundaki isabet sayısı
# HmRun: 1986-1987 sezonundaki en değerli vuruş sayısı
# Runs: 1986-1987 sezonunda takımına kazandırdığı sayı
# RBI: Bir vurucunun vuruş yaptıgında koşu yaptırdığı oyuncu sayısı
# Walks: Karşı oyuncuya yaptırılan hata sayısı
# Years: Oyuncunun major liginde oynama süresi (sene)
# CAtBat: Oyuncunun kariyeri boyunca topa vurma sayısı
# CHits: Oyuncunun kariyeri boyunca yaptığı isabetli vuruş sayısı
# CHmRun: Oyucunun kariyeri boyunca yaptığı en değerli sayısı
# CRuns: Oyuncunun kariyeri boyunca takımına kazandırdığı sayı
# CRBI: Oyuncunun kariyeri boyunca koşu yaptırdırdığı oyuncu sayısı
# CWalks: Oyuncun kariyeri boyunca karşı oyuncuya yaptırdığı hata sayısı
# League: Oyuncunun sezon sonuna kadar oynadığı ligi gösteren A ve N seviyelerine sahip bir faktör
# Division: 1986 sonunda oyuncunun oynadığı pozisyonu gösteren E ve W seviyelerine sahip bir faktör
# PutOuts: Oyun icinde takım arkadaşınla yardımlaşma
# Assits: 1986-1987 sezonunda oyuncunun yaptığı asist sayısı
# Errors: 1986-1987 sezonundaki oyuncunun hata sayısı
# Salary: Oyuncunun 1986-1987 sezonunda aldığı maaş(bin uzerinden)
# NewLeague: 1987 sezonunun başında oyuncunun ligini gösteren A ve N seviyelerine sahip bir faktör



# Importing the libraries
import pandas as pd
import numpy as np

# Importing the dataset
data = pd.read_csv('data/hitters.csv')
data.head()

data.isnull().sum()

data = data.dropna()

data.isnull().sum()


# Categorical and Numerical Columns
categorical_data = data.select_dtypes(include=['object'])
categorical_data.head()
len(categorical_data.columns)
categorical_data.columns

numerical_data = data.select_dtypes(exclude=['object'])
numerical_data.head()
len(numerical_data.columns)
numerical_data.columns


# outlier observation
def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.35)
    quartile3 = dataframe[variable].quantile(0.75)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[(dataframe[col_name] < low_limit), col_name] = low_limit
    dataframe.loc[(dataframe[col_name] > up_limit), col_name] = up_limit

for col in numerical_data.columns:
    print(col, check_outlier(data, col))
    replace_with_thresholds(data, col)



new_num_cols=[col for col in numerical_data if col!="Salary"]

data[new_num_cols]=data[new_num_cols]+0.0000000001

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
data =  pd.get_dummies(data, columns=categorical_data.columns, drop_first=True)
data.head()



# Scale the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data[numerical_data.columns] = scaler.fit_transform(data[numerical_data.columns])

data.head()

# Train-Test Split
from sklearn.model_selection import train_test_split
X = data.drop('Salary', axis=1)
y = data['Salary']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=46)


# Model
from sklearn.linear_model import LinearRegression
lm = LinearRegression()

lm.fit(X_train, y_train)

# Predict
y_pred = lm.predict(X_test)

# Evaluate
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error

print("MSE: ", mean_squared_error(y_test, y_pred))
print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
print("MAE: ", mean_absolute_error(y_test, y_pred))
print("r2_score: ", r2_score(y_test, y_pred))


