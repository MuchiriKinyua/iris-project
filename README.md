![iris flowers](https://github.com/MuchiriKinyua/iris-project/assets/113877377/94e26418-8514-4ee7-b33e-21574a778f9c)
# iris-project
## Table of Contents
1. Business Understanding </br>
   1.1 Business Description </br>
    1.2 Problem Statement </br>
     1.3 Main Objective </br>
       1.4 Specific Objectives
2. Importing Libraries And Warnings
3. Data Understanding
4. Data Preparation
5. EDA
6. Data Preprocessing
7. Data Modelling
8. Evaluation
9. Recommendations
10. Conclusions
 
## 1. Business Understanding
### 1.1 Business Description

### 1.2 Problem Statement
To develop a model that can accurately classify iris flowers into their respective species based on their physical characteristics.

# 1.3 Main Objective
I will build a supervised learning model/models (now that I have the targets) that can learn from the provided dataset containing measurements of iris flowers.

# 1.4 Specific Objectives
### Classification: 
I will build a model that can accurately classify Iris flowers into their three known species (Iris Setosa, Iris Versicolor, Iris Virginica) based on the provided measurements. This will involve learning the patterns i.e The model needs to identify the relationships between the flower measurements (sepal and petal dimensions) and the corresponding species.
### Evaluation:  
A key aspect of evaluation will be to compare the classification accuracy achieved using PCA. This will help me understand if dimensionality reduction through PCA benefits the model performance in this specific case.
## 2. Importing Libraries And Necessary dependencies 
import pickle
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVC
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
from sklearn.decomposition import PCA

## 3. Data Understanding
I got the iris dataset from Kaggle, a popular datasets website.

The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems, and can also be found on the UCI Machine Learning Repository.

It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.

The columns in this dataset are:

Id
SepalLengthCm
SepalWidthCm
PetalLengthCm
PetalWidthCm
Species

## 4. Data Preparation


## 5. EDA

## 6. Data Preprocessing


## 7. Data Modelling


## 8. Evaluation

## 9. Conclusions

   - 
## 10. Recommendations


