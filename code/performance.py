from sklearn.metrics import balanced_accuracy_score, mean_squared_error, roc_auc_score
import csv

def getMetrics(model,X_test,y_test):
    acc = getAccuracy(model,X_test,y_test)
    mse = getMSE(model,X_test,y_test)
    auroc = getAUROC(model,X_test,y_test)

    return

def getAccuracy(model,X_test,y_test):
    y_pred = model.predict(X_test)
    acc = balanced_accuracy_score(y_test,y_pred)
    return acc

def getMSE(model,X_test,y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred,multioutput='uniform_average')
    return mse

def getAUROC():
    y_pred = model.predict_proba(X_test)
    auroc = roc_aoc_score(y_test,y_pred)
    return auroc

def printMetrics(model_name,):

    return

