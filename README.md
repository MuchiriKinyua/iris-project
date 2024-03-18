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
sepal length (cm) Based on the image, the mean seems to be in around 6.0 cm meaning most of the flowers sepal length range around that length </br>
sepal width (cm) For the sepal width on the other hand, it is half the length of sepal length with most of the flowers size ranging at approximately 3.0 </br>
petal length (cm) The length of the petals seem to be unevenly distributed with some ranging from 1 cm to 2.2 cm and from 2.7 cm to 6.8 cm. Also, there is no flower which registered a petal length of between 2.2 cm and 2.7 cm </br>
petal width (cm) Petal width follows the same path with petal length on uneven distribution. Just like sepal width being half the mean of sepal length, petal width is half the size of petal length.

### Bivariate Analysis
sepal length (cm) vs sepal width (cm) - As I said, the sepal length is twice the size of the sepal width. The two seem to have a fairly close distribution. </br>
petal length (cm) vs petal width (cm) - As For the petal length and petal width, business remains the same as that of sepal length and sepal width with both similarity in an evenly distribution and a similar ratio. i.e 2:1 </br>
Each Feature vs Target - 
**N/B: This is an interpretion of where most of the length lies of different features of the each flower** </br>
**sepal length** </br>
setosa - The mean of it's sepal length is 5.0 cm </br>
versicolor - The mean of it's sepal length is 5.7 cm </br>
virginica - The mean of it's sepal length is 6.8 cm </br>

**sepal width** </br>
setosa - The mean of it's sepal width is 2.5 cm </br>
versicolor - The mean of it's sepal width is 2.8 cm </br>
virginica - The mean of it's sepal width is 3.2 cm </br>

**petal length** </br>
setosa - The mean of it's petal length is 1.5 cm </br>
versicolor - The mean of it's petal length is 4.0 cm </br>
virginica - The mean of it's petal length is 5.7 cm </br>

**petal width** </br>
setosa - The mean of it's petal width is 0.25 cm </br>
versicolor - The mean of it's petal width is 1.25 cm </br>
virginica - The mean of it's petal width is 2.0 cm </br>

**target** </br>
setosa (0) - Has 47 features </br>
versicolor (1) - Has 49 features </br>
virginica (2) - Has 50 features </br>

**The analysis above shows that width is half the size of length for each iris species which will be a very important aspect in evaulation**


### Multivariate Analysis 
Petal Width vs. Petal Length for every target </br>
setosa has the smallest petal length and petal width </br>
versicolor has intermediate petal length and petal width as compared to Setosa and virginica </br>
virginica has the largest petal length and petal width

Sepal Width vs. Sepal Length for every target
Just like in petals: setosa has the smallest sepal length and sepal width </br>
versicolor has intermediate sepal length and sepal width as compared to Setosa and virginica </br>
virginica has the largest sepal length and sepal width

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

