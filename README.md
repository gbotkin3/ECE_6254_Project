# ECE_6254_Final_Project

## Abstract
By looking at the visulization of a set of data, is it possible to predict the performance of models relative to each other for the purpose of reliably choosing the best performing model?

## Repository Setup
- ECE_6254__FINAL_PROJECT
  - code/
  - data_files/
  - results/
    - figures/

## Required Dependencies
The following packages are required to run the python file.

1. matplotlib == 3.7.1
2. numpy == 1.24.2
3. pandas == 1.5.3
4. scikit_learn == 1.2.2
5. seaborn == 0.12.2


## Initial Setup

Run the following commands to clone the repository

```
git clone https://github.com/gbotkin3/ECE_6254__Final_Project.git
```
## Running
The python code can be ran from the code directory though the command ```python3 toplevel.py```.

Results are shown in console and stored in ./results and ./results/figures


## Visualization Methods (Stored in ./results/figures)
1. Scatter Map
2. KDE Plot
3. Pair Plot

## Models
1. K-Nearest Neighbors
2. Decision Tree
3. Linear SVC
4. Gaussian Process

## Performance Metrics (Stored in ./results)

1. Balanced Accuracy
2. MSE
3. AUROC
