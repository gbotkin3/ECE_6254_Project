import dataloader as dl
import visualization as vis
import models as mod
import performance as per

import numpy as np

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

# Model Code Here

## Report the Performance of Models

# Performance Code Here