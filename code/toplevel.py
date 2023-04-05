import dataloader as dl
import visualization as vis
import models as mod
import performance as per

import numpy as np

# ignore sklearn FutureWarning
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

## Import the Datasets as Panda Dataframes for visualization and Numpy Arrays for Training and Testing Models

iris_dataset = dl.LoadDataset('iris.csv')        # Iris Dataset in Panda Dataframe Format
iris_dataset = iris_dataset.drop("Id", axis = 1) # Drop the ID Feature as its not a feature used for training.
wheatseeds_dataset = dl.LoadDataset('seeds.csv') # Seeds Dataset in Panda Dataframe Format
drug200_dataset = dl.LoadDataset('drug200.csv')  # Drugs Dataset in Panda Dataframe Format

iris_dataset_numpy = dl.PandaToNumpy(iris_dataset)                                                              # Iris Dataset with Labels in Numpy Array Format
iris_dataset_numpy_labels = np.delete(iris_dataset_numpy, range(0, len(iris_dataset_numpy[0])-1), axis = 1)     # Iris Dataset Labels
iris_dataset_numpy = np.delete(iris_dataset_numpy, -1, axis = 1)                                                # Iris Dataset without Labels in Numpy Array Format
iris_dataset_numpy = dl.scaler(iris_dataset_numpy)                                                              # Iris Dataset in Numpy Array Scaled using sklearn StandardScaler

wheatseeds_dataset_numpy = dl.PandaToNumpy(wheatseeds_dataset)                                                                  # Seeds Dataset with Labels in Numpy Array Format
wheatseeds_dataset_numpy_labels = np.delete(wheatseeds_dataset_numpy, range(0, len(wheatseeds_dataset_numpy[0])-1), axis = 1)   # Seeds Dataset Labels
wheatseeds_dataset_numpy = np.delete(wheatseeds_dataset_numpy, -1, axis = 1)                                                    # Seeds Dataset without Labels in Numpy Array Format
wheatseeds_dataset_numpy = dl.scaler(wheatseeds_dataset_numpy)                                                                  # Seeds Dataset in Numpy Array Scaled using sklearn StandardScaler

drug200_dataset_numpy = dl.GetDummies(drug200_dataset, columns = ["Sex", "BP", "Cholesterol"], prefix = ["Sex", "BP", "Cholesterol"])                                                       # Convert Catagorical Data to Numbers with One Hot Encoding
drug200_dataset_numpy = drug200_dataset_numpy.reindex(columns = ["Age", "Sex_M", "Sex_F", "BP_LOW", "BP_NORMAL", "BP_HIGH", "Cholesterol_NORMAL", "Cholesterol_HIGH", "Na_to_K", "Drug"])   # Reorganize Columns
drug200_dataset_numpy = dl.PandaToNumpy(drug200_dataset_numpy)                                                                                                                              # Drugs Dataset with Labels in Numpy Array Format
drug200_dataset_numpy_labels = np.delete(drug200_dataset_numpy, range(0, len(drug200_dataset_numpy[0])-1), axis = 1)                                                                        # Drugs Dataset Labels
drug200_dataset_numpy = np.delete(drug200_dataset_numpy, -1, axis = 1)                                                                                                                      # Drugs Dataset without Labels in Numpy Array Format
drug200_dataset_numpy = dl.scaler(drug200_dataset_numpy)                                                                                                                                    # Drugs Dataset in Numpy Array Scaled using sklearn StandardScaler

iris_dataset_reduced = dl.ReduceDimensions(iris_dataset_numpy, 3)              #  Iris Dataset Reduced to 3 Dimensions Using PCA
wheatseeds_dataset_reduced = dl.ReduceDimensions(wheatseeds_dataset_numpy, 3)  # Seeds Dataset Reduced to 3 Dimensions Using PCA
drug200_dataset_reduced = dl.ReduceDimensions(drug200_dataset_numpy, 3)        # Drugs Dataset Reduced to 3 Dimensions Using PCA

# Variables Created:

# iris_dataset: Pandas Dataframe Containing the Iris Data
# iris_dataset_numpy: Numpy Array Containing the Iris Data minus the Species
# iris_dataset_numpy_labels: Numpy Array Containg the Iris Data Samples Labels
# iris_dataset_reduced: PCA Reduced Numpy Array Containg Iris Data

# wheatseeds_dataset: Pandas Dataframe Containing the Seeds Data
# wheatseeds_dataset_numpy: Numpy Array Containing the Seeds Data minus the Type
# wheatseeds_dataset_numpy_labels: Numpy Array Containg the Seeds Data Samples Labels
# wheatseeds_dataset_reduced: PCA Reduced Numpy Array Containg Seeds Data

# drug200_dataset: Pandas Dataframe Containing the Drug Data
# drug200_dataset_numpy: Numpy Array Containing the Drug Data minus the Drug Type
# drug200_dataset_numpy_labels: Numpy Array Containg the Drug Data Samples Labels
# drug200_dataset_reduced: PCA Reduced Numpy Array Containg Drug Data

print("Iris Dataset Panda: \n", iris_dataset, "\n Iris Dataset Numpy: \n", iris_dataset_numpy, "\n Iris Dataset Labels: \n", iris_dataset_numpy_labels, "\n Iris Dataset Reduced: \n", iris_dataset_reduced)
print("Seeds Dataset Panda: \n", wheatseeds_dataset, "\n Seeds Dataset Numpy: \n", wheatseeds_dataset_numpy, "\n Seeds Dataset Labels: \n", wheatseeds_dataset_numpy_labels, "\n Seeds Dataset Reduced: \n", wheatseeds_dataset_reduced)
print("Drug Dataset Panda: \n", drug200_dataset, "\n Drug Dataset Numpy: \n", drug200_dataset_numpy, "\n Drug Dataset Labels: \n", drug200_dataset_numpy_labels, "\n Drug Dataset Reduced: \n", drug200_dataset_reduced)

## Visualize the Datasets

# Visual Code Here

## Train and Test Models

# Supervised Models
drug, iris, seeds = mod.loadAllDatasets()
features, targets = mod.splitFeaturesTarget(drug, iris, seeds)

# Pandas To Numpy
for i in range(len(features)):
    features[i] = dl.PandaToNumpy(features[i])
    targets[i] = dl.PandaToNumpy(targets[i])

# Scale Data
scaled_features = []
for i in range(len(features)):
    scaled_features.append(dl.scaler(features[i]))


drugX_train, drugX_test, drugY_train, drugY_test = mod.train_test_split(features[0], targets[0], test_size=0.25)
irisX_train, irisX_test, irisY_train, irisY_test = mod.train_test_split(features[1], targets[1], test_size=0.25)
seedsX_train, seedsX_test, seedsY_train, seedsY_test = mod.train_test_split(features[2], targets[2], test_size=0.25)

# Decision Trees
drug_tree = mod.applyDecisionTree(drugX_train, drugY_train)
iris_tree = mod.applyDecisionTree(irisX_train, irisY_train)
seeds_tree = mod.applyDecisionTree(seedsX_train, seedsY_train)

# SVM
drug_svm = mod.applySVM(drugX_train, drugY_train)
iris_svm = mod.applySVM(irisX_train, irisY_train)
seeds_svm = mod.applySVM(seedsX_train, seedsY_train)

# Gaussian Process
drug_gp = mod.applyGP(drugX_train, drugY_train)
iris_gp = mod.applyGP(irisX_train, irisY_train)
seeds_gp = mod.applyGP(seedsX_train, seedsY_train)

# KNN clf
drug_knn = mod.applyKNN(drugX_train, drugY_train, 15)
iris_knn = mod.applyKNN(irisX_train, irisY_train, 15)
seeds_knn = mod.applyKNN(seedsX_train, seedsY_train, 15)

# TEST
print("--------------- DRUG DATA -------------------")
print("Decision Tree:    ", mod.testModel(drug_tree, drugX_test, drugY_test))
print("SVM:              ", mod.testModel(drug_svm, drugX_test, drugY_test))
print("Gaussian Process: ", mod.testModel(drug_gp, drugX_test, drugY_test))
print("KNN:              ", mod.testModel(drug_knn, drugX_test, drugY_test))

print("--------------- IRIS DATA -------------------")
print("Decision Tree:    ", mod.testModel(iris_tree, irisX_test, irisY_test))
print("SVM:              ", mod.testModel(iris_svm, irisX_test, irisY_test))
print("Gaussian Process: ", mod.testModel(iris_gp, irisX_test, irisY_test))
print("KNN:              ", mod.testModel(iris_knn, irisX_test, irisY_test))

print("--------------- SEEDS DATA ------------------")
print("Decision Tree:    ", mod.testModel(seeds_tree, seedsX_test, seedsY_test))
print("SVM:              ", mod.testModel(seeds_svm, seedsX_test, seedsY_test))
print("Gaussian Process: ", mod.testModel(seeds_gp, seedsX_test, seedsY_test))
print("KNN:              ", mod.testModel(seeds_knn, seedsX_test, seedsY_test))


## Report the Performance of Models

# Decision Trees
#drug_tree_metrics = per.getMetrics(drug_tree,drugX_test,drugY_test)
#iris_tree_metrics = per.getMetrics(iris_tree,irisX_test,irisY_test)
seeds_tree_metrics = per.getMetrics(seeds_tree,seedsX_test,seedsY_test)

# SVM
#drug_svm_metrics = per.getMetrics(drug_svm,drugX_test,drugY_test)
#iris_svm_metrics = per.getMetrics(iris_svm,irisX_test,irisY_test)
seeds_svm_metrics = per.getMetrics(seeds_svm,seedsX_test,seedsY_test)

# Gaussian Process
#drug_gp_metrics = per.getMetrics(drug_gp,drugX_test,drugY_test)
#iris_gp_metrics = per.getMetrics(iris_gp,irisX_test,irisY_test)
seeds_gp_metrics = per.getMetrics(seeds_gp,seedsX_test,seedsY_test)

# KNN
#drug_knn_metrics = per.getMetrics(drug_knn,drugX_test,drugY_test)
#iris_knn_metrics = per.getMetrics(iris_knn,irisX_test,irisY_test)
seeds_knn_metrics = per.getMetrics(seeds_knn,seedsX_test,seedsY_test)

# combine
#drug_metrics = [drug_tree_metrics,drug_svm_metrics,drug_gp_metrics,drug_knn_metrics]
#seeds_metrics = [iris_tree_metrics,iris_svm_metrics,iris_gp_metrics,iris_knn_metrics]
seeds_metrics = [seeds_tree_metrics,seeds_svm_metrics,seeds_gp_metrics,seeds_knn_metrics]

# print
print('\n')
#per.printMetrics('drug',drug_metrics)
#per.printMetrics('iris',seeds_metrics)
per.printMetrics('seeds',seeds_metrics)

# save (to datafiles)
per.exportMetrics(drug_metrics,'drug_metrics.csv')
per.exportMetrics(iris_metrics,'iris_metrics.csv')
per.exportMetrics(seeds_metrics,'seeds_metrics.csv')
