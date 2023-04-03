from dataloader import *
from sklearn.svm import LinearSVC 
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
    svm_clf = LinearSVC()
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

def testModel(model, Xtest, Ytest):
    y_pred = model.predict(Xtest)
    acc = accuracy_score(y_true = Ytest, y_pred = y_pred)
    return acc


# PCA Dim