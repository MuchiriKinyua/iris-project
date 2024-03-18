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
4. Data Cleaning
5. EDA
6. Data Preprocessing
7. Data Modelling
8. Evaluation
9. Conclusions And Recommendations
10. Challenges
 
## 1. Business Understanding
### 1.1 Business Description
Classify iris flowers into three species (setosa, versicolor, virginica) based on 

sepal length, </br> 
sepal width, </br>
petal length, </br>
petal width
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
import pickle </br>
import warnings </br>
import numpy as np </br>
import pandas as pd </br>
import seaborn as sns </br>
from sklearn import metrics </br>
from sklearn.svm import SVC </br>
from sklearn import datasets </br>
import matplotlib.pyplot as plt </br>
warnings.filterwarnings("ignore") </br>
from sklearn.tree import plot_tree </br>
from sklearn.decomposition import PCA </br>
from keras.utils import to_categorical </br>
from keras import models, layers, optimizers </br>
from sklearn.tree import DecisionTreeClassifier </br>
from sklearn.preprocessing import OneHotEncoder </br>
from sklearn.preprocessing import StandardScaler </br>
from sklearn.neighbors import KNeighborsClassifier </br>
from sklearn.linear_model import LogisticRegression </br>
from sklearn.ensemble import RandomForestClassifier </br>
from sklearn.model_selection import train_test_split </br>
from sklearn.metrics import classification_report, accuracy_score

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

## 4. Data Cleaning
**Missing Values** </br>
I started by looking for missing values in my dataset and found none. </br>
**Duplicates** </br>
I also looked for duplicates in my dataset and found none. </br>
**Outliers** </br>
Lastly I looked for outliers in my dataset and found one column, sepal width (cm), to contain 4 outliers. Since this constituted just 2.67% of the data, I decided to drop them since they were significantly small.

## 5. EDA
### Univariate Analysis 
sepal length (cm) </br>
**Observation:** 
**Insight:** 
sepal width (cm) </br>
**Observation:** 
**Insight:** 
petal length (cm) </br>
**Observation:** 
**Insight:** 
petal width (cm)
**Observation:** 
**Insight:** 

### Bivariate Analysis
sepal length (cm) vs sepal width (cm) </br>
**Observation:** 
**Insight:** 
petal length (cm) vs petal width (cm) </br>
**Observation:** 
**Insight:** 
Each Feature vs Target
**Observation:** 
**Insight:** 

### Multivariate Analysis 
Petal Width vs. Petal Length for every target </br>
**Observation:** 
**Insight:**
Sepal Width vs. Sepal Length for every target
**Observation:** 
**Insight:**

## 6. Data Preprocessing
Performing Train Test Split </br>
The dataset was split into training and testing sets before scaling the features. </br>
Scaling </br>
Scaled the features to normalize the range of features in the dataset. </br>
PCA </br>
Created a scatter plot from principal components, where the color of the dot is based on the target value. Separated my variable X_train_pca based on the associated target value in y_train. Created dataframes setosa (target = 0), versicolor (target = 1), and virginica (target = 2).
Correlation </br>
I examined the correlation between numerical columns. I also created a heatmap to visualize the correlation matrix. Notably, a correlation coefficient of approximately 0 indicated no linear relationship between the variables in the dataset.

**Note** The dataset was extensively processed and transformed, resulting in a structured and balanced dataset ready for further analysis and model training. The preprocessing steps aimed to enhance the dataset's quality, reduce dimensionality, setting the stage for effective machine learning model development.

## 7. Data Modelling
used 6 models: </br>
Logistic Regression </br>
Achieved an accuracy of approximately 95%.
SVM </br>
Achieved an accuracy of approximately 94%.
Decision Trees  </br>
Initially trained a Decision Tree model and achieved an accuracy of 1.0.
Conducted Tree pruning to optimize the Decision Tree model.
Achieved an accuracy of 94%.
![Decision tree](https://github.com/MuchiriKinyua/iris-project/assets/113877377/7d1a7524-d819-4a49-b7fc-6db567f1ea87)
Random Forest </br>
Achieved an accuracy of approximately 94%.
KNN </br>
Achieved an accuracy of approximately 94%.
Neural Networks </br>
Achieved an accuracy of approximately 91%.

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

