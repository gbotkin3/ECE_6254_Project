from dataloader import *
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, mean_squared_error, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning
import numpy as np 

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

def applyDecisionTree(Xtrain, Ytrain, max_depth, min_samples_split, min_samples_leaf, criterion, random_state):
    tree_clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, criterion=criterion, random_state=random_state)
    tree_clf = tree_clf.fit(Xtrain, Ytrain)
    return tree_clf

def applyLinearSVC(Xtrain, Ytrain, random_state):
    linearSVC_clf = CalibratedClassifierCV(LinearSVC(random_state=random_state))
    linearSVC_clf = linearSVC_clf.fit(Xtrain, Ytrain)
    return linearSVC_clf

def applyGP(Xtrain, Ytrain, kernel, random_state):
    gp_clf = GaussianProcessClassifier(kernel=kernel, random_state=random_state)
    gp_clf = gp_clf.fit(Xtrain, Ytrain)
    return gp_clf

def applyKNN(Xtrain, Ytrain, n_neighbors):
    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_clf = knn_clf.fit(Xtrain, Ytrain)
    return knn_clf

def applyMLP(Xtrain, Ytrain, alpha, activation, random_state):
    mlp_clf = MLPClassifier(alpha=alpha, activation=activation, random_state=random_state)
    mlp_clf = mlp_clf.fit(Xtrain, Ytrain)
    return mlp_clf

def DecisionTreeTuning(Xtrain, Ytrain, Xtest, Ytest, random_state, debug=False):
    param_grid = {
        "max_depth": [3,5,10,15,20],
        "min_samples_split": [2,5,7,10],
        "min_samples_leaf": [1,2,5],
        "criterion": ['gini', 'entropy']
    }

    clf = DecisionTreeClassifier(random_state=random_state)
    grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc_ovr", n_jobs=-1, cv=3).fit(Xtrain, Ytrain)

    if (debug):
        print("TREE: Param for GS", grid_cv.best_params_)
        print("TREE: CV score for GS", grid_cv.best_score_)
        print("TREE: Train AUC ROC Score for GS: ", roc_auc_score(Ytrain, grid_cv.predict_proba(Xtrain), multi_class='ovr'))
        print("TREE: Test AUC ROC Score for GS: ", roc_auc_score(Ytest, grid_cv.predict_proba(Xtest), multi_class='ovr'))
    return grid_cv.best_params_

def LinearSVCTuning(Xtrain, Ytrain, Xtest, Ytest, random_state, debug=False):
    param_grid = {
        "C": [1,10,100,1000],
        "tol": [1,0.1,0.01,0.001,0.0001, 0.00001],
        "loss": ['squared_hinge'],
        "penalty": ['l2']
    }

    clf = CalibratedClassifierCV(LinearSVC(random_state=random_state))
    grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc_ovr", n_jobs=-1, cv=3).fit(Xtrain, Ytrain)

    if (debug):
        print("SVC: Param for GS", grid_cv.best_params_)
        print("SVC: CV score for GS", grid_cv.best_score_)
        print("SVC: Train AUC ROC Score for GS: ", roc_auc_score(Ytrain, grid_cv.predict(Xtrain), multi_class='ovr'))
        print("SVC: Test AUC ROC Score for GS: ", roc_auc_score(Ytest, grid_cv.predict(Xtest), multi_class='ovr'))

def gpTuning(Xtrain, Ytrain, Xtest, Ytest, random_state, debug=False):
    param_grid = {
        "kernel": [RBF(l) for l in np.logspace(-3.0, 2.0, num=11)],
    }

    clf = GaussianProcessClassifier(random_state=random_state)
    grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc_ovr", n_jobs=None, cv=3).fit(Xtrain, Ytrain)

    if (debug):
        print("GP: Param for GS", grid_cv.best_params_)
        print("GP: CV score for GS", grid_cv.best_score_)
        print("GP: Train AUC ROC Score for GS: ", roc_auc_score(Ytrain, grid_cv.predict_proba(Xtrain), multi_class='ovr'))
        print("GP: Test AUC ROC Score for GS: ", roc_auc_score(Ytest, grid_cv.predict_proba(Xtest), multi_class='ovr'))
    return grid_cv.best_params_

def KNNTuning(Xtrain, Ytrain, Xtest, Ytest, debug=False):
    param_grid = {
        "n_neighbors": [3, 5, 7, 10, 13, 15, 17, 20, 23, 25, 27, 30],
    }

    clf = KNeighborsClassifier()
    grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc_ovr", n_jobs=-1, cv=3).fit(Xtrain, Ytrain)

    if (debug):
        print("KNN: Param for GS", grid_cv.best_params_)
        print("KNN: CV score for GS", grid_cv.best_score_)
        print("KNN: Train AUC ROC Score for GS: ", roc_auc_score(Ytrain, grid_cv.predict_proba(Xtrain), multi_class='ovr'))
        print("KNN: Test AUC ROC Score for GS: ", roc_auc_score(Ytest, grid_cv.predict_proba(Xtest), multi_class='ovr'))
    return grid_cv.best_params_

def MLPtuning(Xtrain, Ytrain, Xtest, Ytest, random_state, debug=False):
    param_grid = {
        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        "alpha":  10.0 ** -np.arange(1, 10),
    }

    clf = MLPClassifier(random_state=random_state)
    grid_cv = GridSearchCV(clf, param_grid, scoring="roc_auc_ovr", n_jobs=None, cv=3).fit(Xtrain, Ytrain)

    if (debug):
        print("MLP: Param for GS", grid_cv.best_params_)
        print("MLP: CV score for GS", grid_cv.best_score_)
        print("MLP: Train AUC ROC Score for GS: ", roc_auc_score(Ytrain, grid_cv.predict_proba(Xtrain), multi_class='ovr'))
        print("MLP: Test AUC ROC Score for GS: ", roc_auc_score(Ytest, grid_cv.predict_proba(Xtest), multi_class='ovr'))
    return grid_cv.best_params_


def testModel(model, Xtest, Ytest):
    y_pred = model.predict(Xtest)
    acc = accuracy_score(y_true = Ytest, y_pred = y_pred)
    return acc
