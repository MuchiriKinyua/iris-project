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
9. Conclusions And Recommendations
10. Challenges
 
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
sepal length (cm)
sepal width (cm)
petal length (cm)
petal width (cm)
target

## 4. Data Preparation


## 5. EDA

## 6. Data Preprocessing


## 7. Data Modelling
used 6 models:
Logistic Regression
SVM
Decision Trees
Random Forest
KNN
Neural Networks

## 8. Evaluation
Ease of Interpretation: Accuracy is an intuitive and easily interpretable metric. It represents the percentage of correct predictions among all predictions made by the model. Users, including hotel staff, can readily understand and trust this metric.

Clear Benchmark: Accuracy provides a clear benchmark for evaluating model performance. It answers the basic question: "How often is the model correct?" This simplicity can be advantageous for communication and decision-making.

Balanced Classes: If the classes of interest (e.g., setosa, versicolor, virginica) are roughly balanced, accuracy can be an effective measure. It doesn't favor one class over another and provides a sense of overall correctness.

I tested and from that we choose the best 3 performing models which are :

Logistic regression = 95%
Random Forest = 94%
KNN = 94%

## 9. Conclusions And Recommendations
1. It provides a good example for exploring various machine learning techniques, particularly classification algorithms.

2. Feature importance analysis: It can provide insights into which features contribute the most to the classification task. This information can be valuable for understanding the underlying characteristics of the data and potentially simplifying the model by focusing on the most relevant features.

3. Outlier detection and removal, as demonstrated in the analysis, can improve model performance by reducing the impact of noisy data points. However, it's essential to exercise caution and consider the domain knowledge when deciding how to handle outliers like I did. But mostly justify.

It's essential to monitor the model's performance over time and periodically retrain it with new data to ensure its effectiveness.
   - 
## 10. Challenges
1. Dataset was too small - Used input method to enable users to input their own sample data

2. Does not demonstrate real world applications due to it's small nature, rather, I used it to simply demonstrate Machine learning concepts

