
import script

path="/Users/chong/Documents/GitHub/Capstone-Project-UCLA/data/train.csv"

def runRFR():
    df = script.loadData(path)
    label, data=script.preprocessing(df)
    selecteddf=script.feature_selection_tree(label, data)
    x_train,x_test,y_train,y_test=script.train_test(selecteddf, label)
    script.RandomForest(x_train,x_test,y_train,y_test)



def runLasso():
    df = script.loadData(path)
    label, data=script.preprocessing(df)
    selecteddf=script.feature_selection_tree(label, data)
    x_train,x_test,y_train,y_test=script.train_test(selecteddf, label)
    script.LinearLasso(x_train,x_test,y_train,y_test)



def runGB():
    df = script.loadData(path)
    label, data=script.preprocessing(df)
    selecteddf=script.feature_selection_tree(label, data)
    x_train,x_test,y_train,y_test=script.train_test(selecteddf, label)
    script.GradientBoost(x_train,x_test,y_train,y_test)


