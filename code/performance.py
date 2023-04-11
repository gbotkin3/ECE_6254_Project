from sklearn.metrics import balanced_accuracy_score, mean_squared_error, roc_auc_score, f1_score, precision_score, recall_score
import csv

def getMetrics(model,X_test,y_test):
    acc = getAccuracy(model,X_test,y_test)
    auroc = getAUROC(model,X_test,y_test)
    f1 = getF1(model,X_test,y_test)
    prec = getPrecision(model,X_test,y_test)
    rec = getRecall(model,X_test,y_test)
    #mse = getMSE(model,X_test,y_test)
    return dict(acc=acc,auroc=auroc,f1=f1,prec=prec,rec=rec)

def getAccuracy(model,X_test,y_test):
    y_pred = model.predict(X_test)
    acc = balanced_accuracy_score(y_test,y_pred)
    return acc

def getAUROC(model,X_test,y_test):
    y_pred_proba = model.predict_proba(X_test)
    auroc = roc_auc_score(y_test,y_pred_proba,multi_class='ovr') # ovr or ovo?
    return auroc

def getF1(model,X_test,y_test):
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test,y_pred,average='macro')
    return f1

def getPrecision(model,X_test,y_test):
    y_pred = model.predict(X_test)
    prec = precision_score(y_test,y_pred,average='macro')
    return prec

def getRecall(model,X_test,y_test):
    y_pred = model.predict(X_test)
    rec = recall_score(y_test,y_pred,average='macro')
    return rec

# def getMSE(model,X_test,y_test):
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test,y_pred,multioutput='uniform_average')
#     return mse

def printMetrics(dataset_name,metrics):
    metric = ['Accuracy','AUROC','F1 Score','Precision','Recall']
    models = ['Decision Trees:','SVM:','Gaussian Process:','KNN:']
    print(f'{f"{dataset_name.upper()} METRICS":-^100}')
    print(f'{"":<10}  {models[0]:^20}  {models[1]:^20}  {models[2]:^20}  {models[3]:^20}')
    for i,key in enumerate(list(metrics[0].keys())):
        print(f'{f"{metric[i]}:":<10}  {metrics[0][key]:<20}  {metrics[1][key]:<20}  {metrics[2][key]:<20}  {metrics[3][key]:<20}')
    return

def exportMetrics(metrics,filename):
    path = '../results/metrics/' + filename
    metric = ['Accuracy','AUROC','F1 Score','Precision','Recall']          # header column
    models = ['','Decision Trees','SVM','Gaussian Process','KNN']          # header row
    metrics = [*[list(x.values()) for x in metrics]]                       # list of dicts to list of lists
    metrics.insert(0,metric)                                               # insert header column
    metrics = [[row[x] for row in metrics] for x in range(len(metrics))]   # swap rows and columns
    metrics.insert(0,models)                                               # insert header row
    with open(path,'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerows(metrics)
