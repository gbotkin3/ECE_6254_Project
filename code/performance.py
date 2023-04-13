from sklearn.metrics import balanced_accuracy_score, mean_squared_error, roc_auc_score
import csv

def getMetrics(model,X_test,y_test):
    acc = getAccuracy(model,X_test,y_test)
    mse = getMSE(model,X_test,y_test)
    auroc = getAUROC(model,X_test,y_test)
    return dict(acc=acc,mse=mse,auroc=auroc)

def getAccuracy(model,X_test,y_test):
    y_pred = model.predict(X_test)
    acc = balanced_accuracy_score(y_test,y_pred)
    return acc

def getMSE(model,X_test,y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test,y_pred,multioutput='uniform_average')
    return mse

def getAUROC(model,X_test,y_test):
    y_pred_proba = model.predict_proba(X_test)
    auroc = roc_auc_score(y_test,y_pred_proba,multi_class='ovr') # ovr or ovo?
    return auroc

def printMetrics(dataset_name,metrics):
    models = ['Decision Trees','SVM','Gaussian Process','KNN']
    print(f'{f"{dataset_name.upper()} METRICS":-^45}')
    for i,model in enumerate(models):
        metric = list(metrics[i].keys())
        print(f'{f"{model}:":<18} Accuracy: {metrics[i][metric[0]]:<19} MSE: {metrics[i][metric[1]]:<5} AUROC: {metrics[i][metric[2]]:<18}')
    return

def exportMetrics(metrics,filename):
    path = '../datafiles/' + filename
    with open(path,'w',newline='') as file:
        writer = csv.DictWriter(file,fieldnames=['acc','mse','auroc'])
        [writer.writerow(x) for x in metrics]
