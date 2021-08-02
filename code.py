import pandas as pd
import numpy as np
from math import sqrt
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier



path="/Users/chong/Documents/GitHub/Capstone-Project-UCLA/data/train.csv"

def loadData(path):
    '''

    load data from csv and drop the Id column

    '''
    # https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
    house_price = pd.read_csv(path)
    df = pd.DataFrame(house_price)

    # Drop id feature
    df = df.drop(["Id"], axis=1)

    return df


def preprocessing(df):
    '''

    dealing with missing value
    one-hot encoding to categorical variables
    variable selection based on feature importance
    training and and validating dataset splitting

    '''
    # Drop column that have over 30% missing values
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

    return label, data



def train_test(df,label):
    '''

    split data into 75% training and 25% testing

    '''
    x_train,x_test,y_train,y_test= train_test_split(df, label, test_size=0.25, random_state=712)
    return x_train,x_test,y_train,y_test


def feature_selection_tree(label, data):
    '''

    use inbuilt class feature_importances of tree based classifiers

    '''
    model = ExtraTreesClassifier()
    model.fit(data, label)
    # print(model.feature_importances_)
    feat_importances = pd.Series(model.feature_importances_, index=data.columns)
    var_list = []
    for key in feat_importances.nlargest(20).keys():
        var_list.append(key)

    selected_df = data[var_list]
    return selected_df


def AICFeatureSelection(label, data):
    '''

    select feature based on linear regression

    '''

    lm = LinearRegression()
    rfe = RFE(lm, n_features_to_select=20)  # running RFEv
    rfe = rfe.fit(data, label)
    var_list = []
    for i in range(len(data.columns)):
        if rfe.support_[i] == True:
            var_list.append(data.columns[i])

    selected_df = data[var_list]
    return selected_df



def RandomForest(x_train,x_test,y_train,y_test):
    '''

    Random Forest Model

    '''
    rf_model = RandomForestRegressor(bootstrap=False, max_depth=500, max_features='auto',
                                     min_samples_leaf=15, criterion='mse', n_jobs=-1, random_state=18).fit(x_train,
                                                                                                           y_train)
    y_pred = rf_model.predict(x_test)
    # cv = cross_val_score(rf_model, x_train, y_train, cv=10)

    rmse = sqrt(mean_squared_error(y_pred, y_test))

    print("The RMSE is :", rmse)




def GradientBoost(x_train,x_test,y_train,y_test):
    '''

    Nerual Network Model

    '''
    # Gradient Boosting
    gbr = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=1, random_state=31).fit(x_train,
                                                                                                            y_train)

    y_pred = gbr.predict(x_test)
    # cv = cross_val_score(rf_model, x_train, y_train, cv=10)

    rmse = sqrt(mean_squared_error(y_pred, y_test))

    print("The RMSE is :", rmse)


def runRFR():
    df = loadData(path)
    label, data=preprocessing(df)
    selecteddf=feature_selection_tree(label, data)
    x_train,x_test,y_train,y_test=train_test(selecteddf, label)
    RandomForest(x_train,x_test,y_train,y_test)


runRFR()