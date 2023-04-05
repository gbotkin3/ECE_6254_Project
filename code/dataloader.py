import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def LoadDataset(path):

    path = "../datafiles/" + path

    dataframe = pd.read_csv(path)

    return dataframe

def PandaToNumpy(dataframe):

    return dataframe.to_numpy()

def GetDummies(dataframe, columns, prefix):

    return pd.get_dummies(dataframe, columns = columns, prefix = prefix)

def scaler(dataset):
    
    scaler = StandardScaler()

    dataset = scaler.fit_transform(dataset)
    
    return dataset

def ReduceDimensions(dataset, n):

    pca = PCA(n_components=n)
    dataset = pca.fit_transform(dataset)

    return dataset