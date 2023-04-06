from dataloader import *
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

# Loading Datasets
def loadAllDatasets():
    drug = LoadDataset("drug200.csv")
    drug = GetDummies(drug, columns = ["Sex", "BP", "Cholesterol"], prefix = None)
    # Move Drug column to the end
    target_col = drug.pop("Drug")
    drug.insert(len(drug.columns), "Drug", target_col)

    iris = LoadDataset("iris.csv")
    iris.pop("Id") # Remove Id Column

    seeds = LoadDataset("seeds.csv")
    return(drug, iris, seeds)

def splitFeaturesTarget(drug, iris, seeds):
    drug_features = drug[drug.columns[0:len(drug.columns) - 1]]
    drug_target = drug[drug.columns[-1]]

    iris_features = iris[iris.columns[0:len(iris.columns) - 1]]
    iris_target = iris[iris.columns[-1]]

    seeds_features = seeds[seeds.columns[0:len(seeds.columns) - 1]]
    seeds_target = seeds[seeds.columns[-1]]
    features = [drug_features, iris_features, seeds_features]
    targets = [drug_target, iris_target, seeds_target]
    return(features, targets)

def applyDecisionTree(Xtrain, Ytrain):
    tree_clf = DecisionTreeClassifier()
    tree_clf = tree_clf.fit(Xtrain, Ytrain)
    return tree_clf

def applySVM(Xtrain, Ytrain):
    svm_clf = CalibratedClassifierCV(LinearSVC())
    svm_clf = svm_clf.fit(Xtrain, Ytrain)
    return svm_clf

def applyGP(Xtrain, Ytrain):
    gp_clf = GaussianProcessClassifier()
    gp_clf = gp_clf.fit(Xtrain, Ytrain)
    return gp_clf

def applyKNN(Xtrain, Ytrain, k):
    knn_clf = KNeighborsClassifier(n_neighbors=k)
    knn_clf = knn_clf.fit(Xtrain, Ytrain)
    return knn_clf

def DecisionTreeTuning(Xtrain, Ytrain, Xtest, Ytest):
    param_grid = {
        "max_depth": [3,5,10,15,20],
        "min_samples_split": [2,5,7,10],
        "min_samples_leaf": [1,2,5],
        "criterion": ['gini', 'entropy']
    }

    clf = DecisionTreeClassifier(random_state=42)
    grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc_ovr", n_jobs=-1, cv=3).fit(Xtrain, Ytrain)

    print("Param for GS", grid_cv.best_params_)
    print("CV score for GS", grid_cv.best_score_)
    print("Train AUC ROC Score for GS: ", roc_auc_score(Ytrain, grid_cv.predict_proba(Xtrain), multi_class='ovr'))
    print("Test AUC ROC Score for GS: ", roc_auc_score(Ytest, grid_cv.predict_proba(Xtest), multi_class='ovr'))
"""
def SVMTuning(Xtrain, Ytrain, Xtest, Ytest):
    param_grid = {
        "C": [3,5,10,15,20],
        "min_samples_split": [2,5,7,10],
        "min_samples_leaf": [1,2,5],
        "criterion": ['gini', 'entropy']
    }

    clf = DecisionTreeClassifier(random_state=42)
    grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc_ovr", n_jobs=-1, cv=3).fit(Xtrain, Ytrain)

    print("Param for GS", grid_cv.best_params_)
    print("CV score for GS", grid_cv.best_score_)
    print("Train AUC ROC Score for GS: ", roc_auc_score(Ytrain, grid_cv.predict_proba(Xtrain), multi_class='ovr'))
    print("Test AUC ROC Score for GS: ", roc_auc_score(Ytest, grid_cv.predict_proba(Xtest), multi_class='ovr'))
"""
def testModel(model, Xtest, Ytest):
    y_pred = model.predict(Xtest)
    acc = accuracy_score(y_true = Ytest, y_pred = y_pred)
    return acc


# PCA Dim
