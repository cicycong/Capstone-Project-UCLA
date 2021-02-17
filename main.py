import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import math
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier

learningRate = 0.01
epoch = 1000
batchSize = 30
numberOfFeatures = 7
reg = 0.01

def dictGen(inputList):
    i = 1
    d = dict()
    for a in inputList:
        d[a] = i
        i = i + 1
    return d

def univariateFeatureSelection(label, data):
    # apply SelectKBest class to rank features
    bestfeatures = SelectKBest(score_func=chi2, k=10)
    fit = bestfeatures.fit(data, label)
    dfscores = pd.DataFrame(fit.scores_)
    dfcolumns = pd.DataFrame(data.columns)
    # concat two dataframes for better visualization
    featureScores = pd.concat([dfcolumns, dfscores], axis=1)
    featureScores.columns = ['Specs', 'Score']  # naming the dataframe columns
    sortFeatures = featureScores.sort_values(by='Score', ascending=False)
    sortFeatures = list(sortFeatures['Specs'])
    # print(sortFeatures)
    # print(featureScores.nlargest(67, 'Score'))  # print 10 best features
    return sortFeatures[:60]

def featureImportance(label, data):
    model = ExtraTreesClassifier()
    model.fit(data, label)
    print(model.feature_importances_)  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=data.columns)
    feat_importances.nlargest(68).plot(kind='barh')
    # plt.show()

def AICFeatureSelection(label, data):
    # Running RFE with the output number of the variable equal to 9
    lm = LinearRegression()
    rfe = RFE(lm, n_features_to_select=20)  # running RFE
    rfe = rfe.fit(data, label)
    print(rfe.support_)  # Printing the boolean results
    print(rfe.ranking_)

def loadData():
    # https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
    dataset = pd.read_csv("dataset/train.csv")
    dataset = pd.DataFrame(dataset)

    # drop all features which has less than 1400 non-NA data.
    # dataset = dataset.dropna(axis=1, thresh=1400, inplace=False)
    # drop id feature
    dataset = dataset.drop(["Id"], axis=1)

    data = pd.DataFrame()

    feature = list(dataset)

    for f in feature:
        if dataset[f].dtype == "object":
            M = dictGen(dataset[f].unique())
            data[f] = dataset[f].map(M).astype("float64")
        elif dataset[f].dtype == "int64":
            # replace nan with average
            if dataset[f].isna().any():
                average = dataset[f].mean()
                dataset[f] = dataset[f].fillna(average, inplace=False)
            data[f] = dataset[f].astype("float64")
        elif dataset[f].dtype == "float64":
            # replace nan with average
            if dataset[f].isna().any():
                average = dataset[f].mean()
                dataset[f] = dataset[f].fillna(average, inplace=False)
        else:
            print("Warning!")

    data.to_csv(path_or_buf="dataset/out.csv")

    # extract label
    label = data['SalePrice']
    data = data.drop(["SalePrice"],axis=1)

    # print the chi scores of each feature
    # select the best features
    selection = univariateFeatureSelection(label, data)

    data = data[selection]

    # plot the importance of features
    # featureImportance(label, data)

    # AIC feature selection
    # AICFeatureSelection(label, data)

    trainData, testData, trainLabel, testLabel = train_test_split(data, label, test_size=0.25, random_state=712)
    return trainData, trainLabel, testData, testLabel

# def plotData():


def buildTensorGraph(inputData):
    tf.set_random_seed(712)

    d1 = 200
    w1 = tf.Variable(tf.random_normal([numberOfFeatures, d1], dtype=tf.float64), dtype=tf.float64)
    b1 = tf.Variable(tf.zeros([d1], dtype=tf.float64), dtype=tf.float64)
    l1 = tf.add(tf.matmul(inputData, w1), b1)

    d2 = 1
    w2 = tf.Variable(tf.random_normal([d1, d2], dtype=tf.float64), dtype=tf.float64)
    b2 = tf.Variable(tf.zeros([d2], dtype=tf.float64), dtype=tf.float64)
    l2 = tf.add(tf.matmul(l1, w2), b2)

    d3 = 1
    w3 = tf.Variable(tf.random_normal([d2, d3], dtype=tf.float64), dtype=tf.float64)
    b3 = tf.Variable(tf.zeros([d3], dtype=tf.float64), dtype=tf.float64)
    l3 = tf.nn.relu(tf.add(tf.matmul(l2, w3), b3))
    return l2, [w1, b1, w2, b2]


def runNN():
    trainData, trainLabel, testData, testLabel = loadData()
    data = tf.placeholder(dtype=tf.float64, shape=(None, numberOfFeatures))
    label = tf.placeholder(dtype=tf.float64, shape=(None, 1))

    output, param = buildTensorGraph(data)

    loss = tf.losses.mean_squared_error(labels=label, predictions=output)

    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for e in range(epoch):
            # pick batch randomly
            trainData, trainLabel = shuffle(trainData, trainLabel)

            for b in range(0, int(trainData.shape[0] / batchSize)):
                batchData = trainData.iloc[b * batchSize:(b + 1) * batchSize, :]
                batchTarget = trainLabel.iloc[b * batchSize:(b + 1) * batchSize]
                # run optimizer
                sess.run(optimizer, feed_dict={label: batchTarget, data: batchData})
            trainLossValue = sess.run(loss, feed_dict={label: trainLabel, data: trainData})
            testLossValue = sess.run(loss, feed_dict={label: testLabel, data: testData})
            print("iteration: ", e, " train loss = ", math.sqrt(trainLossValue), "  test loss = ", math.sqrt(testLossValue))

def runRFR():
    trainData, trainLabel, testData, testLabel = loadData()
    RFR = RandomForestRegressor(n_estimators=1000)
    RFR.fit(trainData, trainLabel)
    testPred = RFR.predict(testData)
    print(np.sum(testPred - testLabel) / len(testLabel))

runRFR()


