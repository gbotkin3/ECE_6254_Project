from sklearn.metrics import balanced_accuracy_score, mean_squared_error, roc_auc_score, f1_score, precision_score, recall_score
import csv

def getMetrics(model,X_test,y_test):
    acc = getAccuracy(model,X_test,y_test)
    auroc = getAUROC(model,X_test,y_test)
    f1 = getF1(model,X_test,y_test)
    prec = getPrecision(model,X_test,y_test)
    rec = getRecall(model,X_test,y_test)
    #mse = getMSE(model,X_test,y_test)
    return [acc,auroc,f1,prec,rec]

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
    f1 = f1_score(y_test,y_pred,average='weighted')
    return f1

def getPrecision(model,X_test,y_test):
    y_pred = model.predict(X_test)
    prec = precision_score(y_test,y_pred,average='weighted')
    return prec

def getRecall(model,X_test,y_test):
    y_pred = model.predict(X_test)
    rec = recall_score(y_test,y_pred,average='weighted')
    return rec

# def getMSE(model,X_test,y_test):
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test,y_pred,multioutput='uniform_average')
#     return mse

def printMetrics(dataset_name,metrics):
    metric = ['Accuracy','AUROC','F1 Score','Precision','Recall']
    model = ['DT:','SVM:','GP:','KNN:','MLP']
    print(f'{f"{dataset_name.upper()} METRICS":-^100}')
    print(f'{"":<5}  {metric[0]:^20}  {metric[1]:^20}  {metric[2]:^20}  {metric[3]:^20}  {metric[4]:^20}')
    for i,m in enumerate(metrics):
        print(f'{f"{model[i]}:":<5}  {metrics[i][0]:<20}  {metrics[i][1]:<20}  {metrics[i][2]:<20}  {metrics[i][3]:<20}  {metrics[i][4]:<20}')
    return

def exportMetrics(metrics,filename):
    path = '../results/metrics/' + filename
    metric = ['','Accuracy','AUROC','F1 Score','Precision','Recall']  # header row
    model = ['DT','SVM','GP','KNN','MLP']                             # header column
    metrics = [[x]+y for x,y in zip(model,metrics)]                   # insert header column
    metrics.insert(0,metric)                                          # insert header row
    with open(path,'w',newline='') as file:
        writer = csv.writer(file)
        writer.writerows(metrics)
