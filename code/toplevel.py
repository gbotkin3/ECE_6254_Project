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
# iris_dataset_numpy = dl.scaler(iris_dataset_numpy)                                                              # Iris Dataset in Numpy Array Scaled using sklearn StandardScaler

wheatseeds_dataset_numpy = dl.PandaToNumpy(wheatseeds_dataset)                                                                  # Seeds Dataset with Labels in Numpy Array Format
wheatseeds_dataset_numpy_labels = np.delete(wheatseeds_dataset_numpy, range(0, len(wheatseeds_dataset_numpy[0])-1), axis = 1)   # Seeds Dataset Labels
wheatseeds_dataset_numpy = np.delete(wheatseeds_dataset_numpy, -1, axis = 1)                                                    # Seeds Dataset without Labels in Numpy Array Format
# wheatseeds_dataset_numpy = dl.scaler(wheatseeds_dataset_numpy)                                                                  # Seeds Dataset in Numpy Array Scaled using sklearn StandardScaler

drug200_dataset_numpy = dl.GetDummies(drug200_dataset, columns = ["Sex", "BP", "Cholesterol"], prefix = ["Sex", "BP", "Cholesterol"])                                                       # Convert Catagorical Data to Numbers with One Hot Encoding
drug200_dataset_numpy = drug200_dataset_numpy.reindex(columns = ["Age", "Sex_M", "Sex_F", "BP_LOW", "BP_NORMAL", "BP_HIGH", "Cholesterol_NORMAL", "Cholesterol_HIGH", "Na_to_K", "Drug"])   # Reorganize Columns
drug200_dataset_numpy = dl.PandaToNumpy(drug200_dataset_numpy)                                                                                                                              # Drugs Dataset with Labels in Numpy Array Format
drug200_dataset_numpy_labels = np.delete(drug200_dataset_numpy, range(0, len(drug200_dataset_numpy[0])-1), axis = 1)                                                                        # Drugs Dataset Labels
drug200_dataset_numpy = np.delete(drug200_dataset_numpy, -1, axis = 1)                                                                                                                      # Drugs Dataset without Labels in Numpy Array Format
# drug200_dataset_numpy = dl.scaler(drug200_dataset_numpy)                                                                                                                                    # Drugs Dataset in Numpy Array Scaled using sklearn StandardScaler

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

# print("Iris Dataset Panda: \n", iris_dataset, "\n Iris Dataset Numpy: \n", iris_dataset_numpy, "\n Iris Dataset Labels: \n", iris_dataset_numpy_labels, "\n Iris Dataset Reduced: \n", iris_dataset_reduced)
# print("Seeds Dataset Panda: \n", wheatseeds_dataset, "\n Seeds Dataset Numpy: \n", wheatseeds_dataset_numpy, "\n Seeds Dataset Labels: \n", wheatseeds_dataset_numpy_labels, "\n Seeds Dataset Reduced: \n", wheatseeds_dataset_reduced)
# print("Drug Dataset Panda: \n", drug200_dataset, "\n Drug Dataset Numpy: \n", drug200_dataset_numpy, "\n Drug Dataset Labels: \n", drug200_dataset_numpy_labels, "\n Drug Dataset Reduced: \n", drug200_dataset_reduced)

## Visualize the Datasets

# Visual Code Here

## Train and Test Models

drug200_dataset_numpy_labels = np.ravel(drug200_dataset_numpy_labels, order='C')
iris_dataset_numpy_labels = np.ravel(iris_dataset_numpy_labels, order='C')
wheatseeds_dataset_numpy_labels = np.ravel(wheatseeds_dataset_numpy_labels, order='C')
# Normal Datasets
drugX_train, drugX_test, drugY_train, drugY_test = mod.train_test_split(drug200_dataset_numpy, drug200_dataset_numpy_labels, test_size=0.25)
irisX_train, irisX_test, irisY_train, irisY_test = mod.train_test_split(iris_dataset_numpy, iris_dataset_numpy_labels, test_size=0.25)
seedsX_train, seedsX_test, seedsY_train, seedsY_test = mod.train_test_split(wheatseeds_dataset_numpy, wheatseeds_dataset_numpy_labels, test_size=0.25)
#Reduced Datasets
drugX_train_reduced, drugX_test_reduced, drugY_train_reduced, drugY_test_reduced = mod.train_test_split(drug200_dataset_reduced, drug200_dataset_numpy_labels, test_size=0.25)
irisX_train_reduced, irisX_test_reduced, irisY_train_reduced, irisY_test_reduced = mod.train_test_split(iris_dataset_reduced, iris_dataset_numpy_labels, test_size=0.25)
seedsX_train_reduced, seedsX_test_reduced, seedsY_train_reduced, seedsY_test_reduced = mod.train_test_split(wheatseeds_dataset_reduced, wheatseeds_dataset_numpy_labels, test_size=0.25)

# Scale
drugX_train = dl.scaler(drugX_train)
drugX_test = dl.scaler(drugX_test)
irisX_train = dl.scaler(irisX_train)
irisX_test = dl.scaler(irisX_test)
seedsX_train = dl.scaler(seedsX_train) 
seedsX_test = dl.scaler(seedsX_test)

drugX_train_reduced = dl.scaler(drugX_train_reduced)
drugX_test_reduced = dl.scaler(drugX_test_reduced)
irisX_train_reduced = dl.scaler(irisX_train_reduced)
irisX_test_reduced = dl.scaler(irisX_test_reduced)
seedsX_train_reduced = dl.scaler(seedsX_train_reduced) 
seedsX_test_reduced = dl.scaler(seedsX_test_reduced)


# Decision Trees
drug_tree_params = mod.DecisionTreeTuning(drugX_train, drugY_train, drugX_test, drugY_test)
drug_knn_params = mod.KNNTuning(drugX_train, drugY_train, drugX_test, drugY_test)
drug_gp_params = mod.gpTuning(drugX_train, drugY_train, drugX_test, drugY_test)
iris_tree_params = mod.DecisionTreeTuning(irisX_train, irisY_train, irisX_test, irisY_test)
iris_knn_params = mod.KNNTuning(irisX_train, irisY_train, irisX_test, irisY_test)
iris_gp_params = mod.gpTuning(irisX_train, irisY_train, irisX_test, irisY_test)
seeds_tree_params = mod.DecisionTreeTuning(seedsX_train, seedsY_train, seedsX_test, seedsY_test)
seeds_knn_params = mod.KNNTuning(seedsX_train, seedsY_train, seedsX_test, seedsY_test)
seeds_gp_params = mod.gpTuning(seedsX_train, seedsY_train, seedsX_test, seedsY_test)

red_drug_tree_params = mod.DecisionTreeTuning(drugX_train_reduced, drugY_train_reduced, drugX_test_reduced, drugY_test_reduced)
red_drug_knn_params = mod.KNNTuning(drugX_train_reduced, drugY_train_reduced, drugX_test_reduced, drugY_test_reduced)
red_drug_gp_params = mod.gpTuning(drugX_train_reduced, drugY_train_reduced, drugX_test_reduced, drugY_test_reduced)
red_iris_tree_params = mod.DecisionTreeTuning(irisX_train_reduced, irisY_train_reduced, irisX_test_reduced, irisY_test_reduced)
red_iris_knn_params = mod.KNNTuning(irisX_train_reduced, irisY_train_reduced, irisX_test_reduced, irisY_test_reduced)
red_iris_gp_params = mod.gpTuning(irisX_train_reduced, irisY_train_reduced, irisX_test_reduced, irisY_test_reduced)
red_seeds_tree_params = mod.DecisionTreeTuning(seedsX_train_reduced, seedsY_train_reduced, seedsX_test_reduced, seedsY_test_reduced)
red_seeds_knn_params = mod.KNNTuning(seedsX_train_reduced, seedsY_train_reduced, seedsX_test_reduced, seedsY_test_reduced)
red_seeds_gp_params = mod.gpTuning(seedsX_train_reduced, seedsY_train_reduced, seedsX_test_reduced, seedsY_test_reduced)

drug_tree = mod.applyDecisionTree(drugX_train, drugY_train, drug_tree_params['max_depth'], drug_tree_params['min_samples_split'], drug_tree_params['min_samples_leaf'], drug_tree_params['criterion'])
iris_tree = mod.applyDecisionTree(irisX_train, irisY_train, iris_tree_params['max_depth'], iris_tree_params['min_samples_split'], iris_tree_params['min_samples_leaf'], iris_tree_params['criterion'])
seeds_tree = mod.applyDecisionTree(seedsX_train, seedsY_train, seeds_tree_params['max_depth'], seeds_tree_params['min_samples_split'], seeds_tree_params['min_samples_leaf'], seeds_tree_params['criterion'])
drug_tree_reduced = mod.applyDecisionTree(drugX_train_reduced, drugY_train_reduced, red_drug_tree_params['max_depth'], red_drug_tree_params['min_samples_split'], red_drug_tree_params['min_samples_leaf'], red_drug_tree_params['criterion'])
iris_tree_reduced = mod.applyDecisionTree(irisX_train_reduced, irisY_train_reduced, red_iris_tree_params['max_depth'], red_iris_tree_params['min_samples_split'], red_iris_tree_params['min_samples_leaf'], red_iris_tree_params['criterion'])
seeds_tree_reduced = mod.applyDecisionTree(seedsX_train_reduced, seedsY_train_reduced, red_seeds_tree_params['max_depth'], red_seeds_tree_params['min_samples_split'], red_seeds_tree_params['min_samples_leaf'], red_seeds_tree_params['criterion'])

# SVM
drug_svm = mod.applyLinearSVC(drugX_train, drugY_train)
iris_svm = mod.applyLinearSVC(irisX_train, irisY_train)
seeds_svm = mod.applyLinearSVC(seedsX_train, seedsY_train)
drug_svm_reduced = mod.applyLinearSVC(drugX_train_reduced, drugY_train_reduced)
iris_svm_reduced = mod.applyLinearSVC(irisX_train_reduced, irisY_train_reduced)
seeds_svm_reduced = mod.applyLinearSVC(seedsX_train_reduced, seedsY_train_reduced)

# Gaussian Process
drug_gp = mod.applyGP(drugX_train, drugY_train, drug_gp_params['kernel'])
iris_gp = mod.applyGP(irisX_train, irisY_train, iris_gp_params['kernel'])
seeds_gp = mod.applyGP(seedsX_train, seedsY_train, seeds_gp_params['kernel'])
drug_gp_reduced = mod.applyGP(drugX_train_reduced, drugY_train_reduced, red_drug_gp_params['kernel'])
iris_gp_reduced = mod.applyGP(irisX_train_reduced, irisY_train_reduced, red_iris_gp_params['kernel'])
seeds_gp_reduced = mod.applyGP(seedsX_train_reduced, seedsY_train_reduced, red_seeds_gp_params['kernel'])

# KNN clf
drug_knn = mod.applyKNN(drugX_train, drugY_train, drug_knn_params['n_neighbors'])
iris_knn = mod.applyKNN(irisX_train, irisY_train, iris_knn_params['n_neighbors'])
seeds_knn = mod.applyKNN(seedsX_train, seedsY_train, seeds_knn_params['n_neighbors'])
drug_knn_reduced = mod.applyKNN(drugX_train_reduced, drugY_train_reduced, red_drug_knn_params['n_neighbors'])
iris_knn_reduced = mod.applyKNN(irisX_train_reduced, irisY_train_reduced, red_iris_knn_params['n_neighbors'])
seeds_knn_reduced = mod.applyKNN(seedsX_train_reduced, seedsY_train_reduced, red_seeds_knn_params['n_neighbors'])

# TEST
print("--------------- DRUG DATA -------------------")
print("Decision Tree:    ", mod.testModel(drug_tree, drugX_test, drugY_test))
print("SVM:              ", mod.testModel(drug_svm, drugX_test, drugY_test))
print("Gaussian Process: ", mod.testModel(drug_gp, drugX_test, drugY_test))
print("KNN:              ", mod.testModel(drug_knn, drugX_test, drugY_test))
print("--------------- REDUCED DRUG DATA -------------------")
print("Decision Tree:    ", mod.testModel(drug_tree_reduced, drugX_test_reduced, drugY_test_reduced))
print("SVM:              ", mod.testModel(drug_svm_reduced, drugX_test_reduced, drugY_test_reduced))
print("Gaussian Process: ", mod.testModel(drug_gp_reduced, drugX_test_reduced, drugY_test_reduced))
print("KNN:              ", mod.testModel(drug_knn_reduced, drugX_test_reduced, drugY_test_reduced))

print("--------------- IRIS DATA -------------------")
print("Decision Tree:    ", mod.testModel(iris_tree, irisX_test, irisY_test))
print("SVM:              ", mod.testModel(iris_svm, irisX_test, irisY_test))
print("Gaussian Process: ", mod.testModel(iris_gp, irisX_test, irisY_test))
print("KNN:              ", mod.testModel(iris_knn, irisX_test, irisY_test))
print("--------------- REDUCED IRIS DATA -------------------")
print("Decision Tree:    ", mod.testModel(iris_tree_reduced, irisX_test_reduced, irisY_test_reduced))
print("SVM:              ", mod.testModel(iris_svm_reduced, irisX_test_reduced, irisY_test_reduced))
print("Gaussian Process: ", mod.testModel(iris_gp_reduced, irisX_test_reduced, irisY_test_reduced))
print("KNN:              ", mod.testModel(iris_knn_reduced, irisX_test_reduced, irisY_test_reduced))

print("--------------- SEEDS DATA ------------------")
print("Decision Tree:    ", mod.testModel(seeds_tree, seedsX_test, seedsY_test))
print("SVM:              ", mod.testModel(seeds_svm, seedsX_test, seedsY_test))
print("Gaussian Process: ", mod.testModel(seeds_gp, seedsX_test, seedsY_test))
print("KNN:              ", mod.testModel(seeds_knn, seedsX_test, seedsY_test))
print("--------------- REDUCED SEEDS DATA ------------------")
print("Decision Tree:    ", mod.testModel(seeds_tree_reduced, seedsX_test_reduced, seedsY_test_reduced))
print("SVM:              ", mod.testModel(seeds_svm_reduced, seedsX_test_reduced, seedsY_test_reduced))
print("Gaussian Process: ", mod.testModel(seeds_gp_reduced, seedsX_test_reduced, seedsY_test_reduced))
print("KNN:              ", mod.testModel(seeds_knn_reduced, seedsX_test_reduced, seedsY_test_reduced))


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
