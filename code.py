import pandas as pd
from math import sqrt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

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

#
#
# def nn_model(lr=0.01):
#     '''
#
#     Neural Network Model
#     '''
#     model = Sequential()
#
#     model.add(Dense(units=1000, activation='relu'))
#     model.add(Dropout(0.5))
#
#     model.add(Dense(units=50, activation='relu'))
#     model.add(Dropout(0.5))
#
#     model.add(Dense(units=3, activation='softmax'))
#
#     opt=tf.keras.optimizers.Adam(learning_rate=0.01)
#
#     model.compile(optimizer='rmsprop',
#                   loss='mse',
#                   metrics=['mse'])
#
#     return model
#
#
# def buildTensorGraph(inputData):
#     tf.set_random_seed(712)
#
#     d1 = 200
#     w1 = tf.Variable(tf.random_normal([numberOfFeatures, d1], dtype=tf.float64), dtype=tf.float64)
#     b1 = tf.Variable(tf.zeros([d1], dtype=tf.float64), dtype=tf.float64)
#     l1 = tf.add(tf.matmul(inputData, w1), b1)
#
#     d2 = 1
#     w2 = tf.Variable(tf.random_normal([d1, d2], dtype=tf.float64), dtype=tf.float64)
#     b2 = tf.Variable(tf.zeros([d2], dtype=tf.float64), dtype=tf.float64)
#     l2 = tf.add(tf.matmul(l1, w2), b2)
#
#     d3 = 1
#     w3 = tf.Variable(tf.random_normal([d2, d3], dtype=tf.float64), dtype=tf.float64)
#     b3 = tf.Variable(tf.zeros([d3], dtype=tf.float64), dtype=tf.float64)
#     l3 = tf.nn.relu(tf.add(tf.matmul(l2, w3), b3))
#     return l2, [w1, b1, w2, b2]


def runRFR():
    df = loadData(path)
    label, data=preprocessing(df)
    selecteddf=feature_selection_tree(label, data)
    x_train,x_test,y_train,y_test=train_test(selecteddf, label)
    RandomForest(x_train,x_test,y_train,y_test)



#
#
# def runNN2():
#     df = loadData(path)
#     label, data=preprocessing(df)
#     selecteddf=feature_selection_tree(label, data)
#     x_train,x_test,y_train,y_test=train_test(selecteddf, label)
#     model = nn_model()
#     model.fit(x=X_train, y=y_train, batch_size=20, epochs=10, verbose=1)
#     y_prednn = model.predict(X_test)
#
#     rmse = sqrt(mean_squared_error(y_prednn, y_test))
#
#     print("The RMSE is :", rmse)
#
# def runNN():
#
#
#     df = loadData(path)
#     label, data=preprocessing(df)
#     selecteddf=feature_selection_tree(label, data)
#     x_train, x_test, y_train, y_test = train_test(selecteddf, label)
#
#
#     data = tf.placeholder(dtype=tf.float64, shape=(None, numberOfFeatures))
#     label = tf.placeholder(dtype=tf.float64, shape=(None, 1))
#
#     output, param = buildTensorGraph(data)
#
#     loss = tf.losses.mean_squared_error(labels=label, predictions=output)
#
#     optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)
#
#     init_op = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init_op)
#         for e in range(epoch):
#             # pick batch randomly
#             x_train, x_test = shuffle(x_train, x_test)
#
#             for b in range(0, int(x_train.shape[0] / batchSize)):
#                 batchData = x_train.iloc[b * batchSize:(b + 1) * batchSize, :]
#                 batchTarget = x_test.iloc[b * batchSize:(b + 1) * batchSize]
#                 # run optimizer
#                 sess.run(optimizer, feed_dict={label: batchTarget, data: batchData})
#             trainLossValue = sess.run(loss, feed_dict={label: x_test, data: x_train})
#             testLossValue = sess.run(loss, feed_dict={label: y_test, data: testData})
#             print("iteration: ", e, " train loss = ", math.sqrt(trainLossValue), "  test loss = ", math.sqrt(testLossValue))
#
# # runNN2()
runRFR()