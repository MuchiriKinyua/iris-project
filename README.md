![iris flowers](https://github.com/MuchiriKinyua/iris-project/assets/113877377/94e26418-8514-4ee7-b33e-21574a778f9c)
# iris-project 
## 1. Business Understanding
### 1.1 Business Description
Classify iris flowers into three species (setosa, versicolor, virginica) based on 

sepal length, </br> 
sepal width, </br>
petal length, </br>
petal width
### 1.2 Problem Statement
Classifying flowers can be challenging, especially with similarities in their morphological features. In botanical studies and horticulture, accurately identifying the species of a flower is crucial for various purposes, such as species conservation, breeding programs, and understanding ecological dynamics. However, manual classification based on morphological features can be time-consuming and subjective, leading to inconsistencies in identification.

To address this challenge, the project aims to develop a machine learning model that can accurately classify flowers based on their morphological characteristics. By leveraging a dataset containing measurements of several features of different flower species, such as sepal length, sepal width, petal length, and petal width, the project seeks to train a model capable of distinguishing between species with a high degree of accuracy.

# 1.3 Main Objective
I will build a supervised learning model/models (now that I have the targets) that can learn from the provided dataset containing measurements of iris flowers.

# 1.4 Specific Objectives
### Classification: 
I will build a model that can accurately classify Iris flowers into their three known species (Iris Setosa, Iris Versicolor, Iris Virginica) based on the provided measurements. This will involve learning the patterns i.e The model needs to identify the relationships between the flower measurements (sepal and petal dimensions) and the corresponding species.
### Evaluation:  
A key aspect of evaluation will be to compare the classification accuracy achieved using PCA. This will help me understand if dimensionality reduction through PCA benefits the model performance in this specific case.

## 2. Data Understanding
I got the iris dataset from Kaggle, a popular datasets website.

The Iris dataset was used in R.A. Fisher's classic 1936 paper, The Use of Multiple Measurements in Taxonomic Problems, and can also be found on the UCI Machine Learning Repository.

It includes three iris species with 50 samples each as well as some properties about each flower. One flower species is linearly separable from the other two, but the other two are not linearly separable from each other.

The columns in this dataset are: </br>

Id </br>
sepal length (cm) </br>
sepal width (cm) </br>
petal length (cm) </br>
petal width (cm) </br>
target

## 3. Data Cleaning
**Missing Values** </br>
I started by looking for missing values in my dataset and found none. </br>
**Duplicates** </br>
I also looked for duplicates in my dataset and found none. </br>
**Outliers** </br>
Lastly I looked for outliers in my dataset and found one column, sepal width (cm), to contain 4 outliers. Since this constituted just 2.67% of the data, I decided to drop them since they were significantly small.

## 4. EDA
### Univariate Analysis 
![sepal length (cm)](https://github.com/MuchiriKinyua/iris-project/assets/113877377/2878fc1c-2c0e-40e1-8d9a-1dead39b2f70)
sepal length (cm) Based on the image, the mean seems to be in around 6.0 cm meaning most of the flowers sepal length range around that length </br>
![Sepal width](https://github.com/MuchiriKinyua/iris-project/assets/113877377/9608a875-1ae6-48f7-a70c-dd25eb3fad55)
sepal width (cm) For the sepal width on the other hand, it is half the length of sepal length with most of the flowers size ranging at approximately 3.0 </br>
![Petal length](https://github.com/MuchiriKinyua/iris-project/assets/113877377/5592c64b-6ccd-47dc-813e-0c350b6f413f)
petal length (cm) The length of the petals seem to be unevenly distributed with some ranging from 1 cm to 2.2 cm and from 2.7 cm to 6.8 cm. Also, there is no flower which registered a petal length of between 2.2 cm and 2.7 cm </br>
![Petal width](https://github.com/MuchiriKinyua/iris-project/assets/113877377/c5e13746-ef7f-45df-afb4-e8daaf6e1f22)
petal width (cm) Petal width follows the same path with petal length on uneven distribution. Just like sepal width being half the mean of sepal length, petal width is half the size of petal length.

### Bivariate Analysis
![sepal length vs sepal width png ](https://github.com/MuchiriKinyua/iris-project/assets/113877377/2d771e85-aa0f-48cd-a65b-ce9ac6411bea)
sepal length (cm) vs sepal width (cm) - As I said, the sepal length is twice the size of the sepal width. The two seem to have a fairly close distribution. </br>
![Petal length vs petal width](https://github.com/MuchiriKinyua/iris-project/assets/113877377/67b8d5dc-5fdf-4047-af38-5a11b333b553)
petal length (cm) vs petal width (cm) - As For the petal length and petal width, business remains the same as that of sepal length and sepal width with both similarity in an evenly distribution and a similar ratio. i.e 2:1 </br>
Each Feature vs Target - </br>
![Feature and Target](https://github.com/MuchiriKinyua/iris-project/assets/113877377/0109b2c6-9481-409a-99cc-80e77f725efd)
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
![Multivariate petal png ](https://github.com/MuchiriKinyua/iris-project/assets/113877377/648c67a4-d65c-4105-b47c-66bc4fd971a4)
Petal Width vs. Petal Length for every target </br>
setosa has the smallest petal length and petal width </br>
versicolor has intermediate petal length and petal width as compared to Setosa and virginica </br>
virginica has the largest petal length and petal width

Sepal Width vs. Sepal Length for every target
![Multivariate sepal](https://github.com/MuchiriKinyua/iris-project/assets/113877377/11abaf3c-ed9d-4e9f-8111-b8ad4c081e24)
Just like in petals: setosa has the smallest sepal length and sepal width </br>
versicolor has intermediate sepal length and sepal width as compared to Setosa and virginica </br>
virginica has the largest sepal length and sepal width

## 5. Data Preprocessing
**Performing Train Test Split** </br>
The dataset was split into training and testing sets before scaling the features. </br>
**Scaling** </br>
Scaled the features to normalize the range of features in the dataset. </br>
**PCA** </br>
Created a scatter plot from principal components, where the color of the dot is based on the target value. Separated my variable X_train_pca based on the associated target value in y_train. Created dataframes setosa (target = 0),  </br> versicolor (target = 1),  </br> virginica (target = 2). </br>
**Correlation** </br>
I examined the correlation between numerical columns. I also created a heatmap to visualize the correlation matrix. Notably, a correlation coefficient of approximately 0 indicated no linear relationship between the variables in the dataset.

**Note** The dataset was extensively processed and transformed, resulting in a structured and balanced dataset ready for further analysis and model training. The preprocessing steps aimed to enhance the dataset's quality, reduce dimensionality, setting the stage for effective machine learning model development.

## 6. Data Modelling
Used 6 models: </br>
**Logistic Regression** </br>
Achieved an accuracy of approximately 95%. </br>
**SVM** </br>
Achieved an accuracy of approximately 94%. </br>
**Decision Trees**  </br>
Initially trained a Decision Tree model and achieved an accuracy of 1.0.
Conducted Tree pruning to optimize the Decision Tree model.
Achieved an accuracy of 94%. </br>
**Random Forest** </br>
Achieved an accuracy of approximately 94%. </br>
**KNN** </br>
Achieved an accuracy of approximately 94%. </br>
**Neural Networks** </br>
Achieved an accuracy of approximately 91%. </br>

## 7. Evaluation
Ease of Interpretation: Accuracy is an intuitive and easily interpretable metric. It represents the percentage of correct predictions among all predictions made by the model. Users, including hotel staff, can readily understand and trust this metric.

Clear Benchmark: Accuracy provides a clear benchmark for evaluating model performance. It answers the basic question: "How often is the model correct?" This simplicity can be advantageous for communication and decision-making.

Balanced Classes: If the classes of interest (e.g., setosa, versicolor, virginica) are roughly balanced, accuracy can be an effective measure. It doesn't favor one class over another and provides a sense of overall correctness.

I tested and from that we choose the best 3 performing models which are : </br>

Logistic regression = 95% </br>
Random Forest = 94% </br>
KNN = 94% </br>

## 8. Conclusions And Recommendations
1. It provides a good example for exploring various machine learning techniques, particularly classification algorithms.

2. Feature importance analysis: It can provide insights into which features contribute the most to the classification task. This information can be valuable for understanding the underlying characteristics of the data and potentially simplifying the model by focusing on the most relevant features.

3. Outlier detection and removal, as demonstrated in the analysis, can improve model performance by reducing the impact of noisy data points. However, it's essential to exercise caution and consider the domain knowledge when deciding how to handle outliers like I did. But mostly justify.
4. Petal length and sepal length are twice the size of petal width and sepal width.
5. It's essential to monitor the model's performance over time and periodically retrain it with new data to ensure its effectiveness.
   
## 9. Challenges
1. Dataset was too small - Used input method to enable users to input their own sample data

2. Does not demonstrate real world applications due to it's small nature, rather, I used it to simply demonstrate Machine learning concepts

