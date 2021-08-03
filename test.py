import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

# def loadData():
    # https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview

path="/Users/chong/Documents/GitHub/Capstone-Project-UCLA/data/train.csv"

house_price = pd.read_csv(path)
df = pd.DataFrame(house_price)

# Drop id feature
df = df.drop(["Id"], axis=1)

df = df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu'])

# missing value imputation
# impute numeric missing variables with median value
# impute categorical missing variables with the most occured class
for f in df.columns:
    if df[f].dtype == "object":
        df[f] = df[f].fillna(df[f].value_counts().index[0], inplace=False)
    elif df[f].dtype == "int64":
        # replace nan with average
        if df[f].isna().any():
            df[f] = df[f].fillna(df[f].median(), inplace=False)
        df[f] = df[f].astype("float64")
    elif df[f].dtype == "float64":
        # replace nan with average
        if df[f].isna().any():
            df[f] = df[f].fillna(df[f].median(), inplace=False)
    else:
        print("Warning!")

# one-hot encoding
cat_var = df.select_dtypes(include=["object"]).columns
df = pd.get_dummies(df, columns=cat_var, drop_first=True)

label = df['SalePrice']
data = df.drop(["SalePrice"], axis=1)

model = ExtraTreesClassifier()
model.fit(data, label)
# print(model.feature_importances_)
feat_importances = pd.Series(model.feature_importances_, index=data.columns)
var_list = []
for key in feat_importances.nlargest(20).keys():
    var_list.append(key)

df = df[var_list]

x_train,x_test,y_train,y_test= train_test_split(df, label, test_size=0.25, random_state=712)

# rf_model = RandomForestRegressor(bootstrap=False, max_depth=500, max_features='auto',
#                                  min_samples_leaf=15, criterion='mse', n_jobs=-1, random_state=18).fit(x_train,
#                                                                                                        y_train)
# y_pred = rf_model.predict(x_test)
#
#
# rmse = np.sum(y_pred - y_test) / len(y_test)
#
# print(rmse)

model = Sequential()

model.add(Dense(units=1000, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=50, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=3, activation='softmax'))

opt = tf.keras.optimizers.Adam(learning_rate=0.01)

model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['mse'])

model.fit(x=X_train, y=y_train, batch_size=20, epochs=10, verbose=1)
y_prednn = model.predict(X_test)

rmse = sqrt(mean_squared_error(y_prednn, y_test))

print("The RMSE is :", rmse)
