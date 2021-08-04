# [tbrain-2019 國泰大數據競賽](https://tbrain.trendmicro.com.tw/Competitions/Details/7)
#### Public Leaderboard:  10 / 244 (score: 0.850655)
#### Private Leaderboard: 26 / 244 (score: 0.845608)<br>

## Task
#### Supervised learning for binary classification<br>

## Data description
#### We remove the "customer number field" and "target variable field" from the original data, and slightly classify the remaining fields according to their value types to facilitate subsequent data processing.

#### Train：# 100,000 ( Y：# 2,000 / N：# 98,000)
#### Test ：# 150,000<br>

|      Column type       |             Example              | Number of fields |
|:----------------------:|:--------------------------------:|:----------------:|
| categorical (ordinal)  | {Low, medium, medium high, high} |        #  6      |
| categorical (logical)  | {Y, N}                           |        # 79      |
| categorical (nominal)  | {Male, Female}                   |        #  4      |
| numerical (continuous) | {0.125, 0.375, 0}                |        # 21      |
| numerical (discrete)   | {0, 1, 2, 3}                     |        # 20      |
<br>

## Data cleaning
#### Categorical columns
> Ordinary features  
>> * Replace accordingly with integers from small to large.  
>> * The part of the empty value is filled with "zero value" because it cannot bring comparable information.  

> Logical features  
>> * Replace the binary feature with the content of Y/N with the value 1/0.  
>> * Since the value "0" here already has a representative message, we use the value "2" to fill in the blank value.  

> Nominal features  
>> * Use dummy variables for conversion (empty values are also classified as a category).  

#### Numerical columns  
> Outliers  
>> * Use [Q1−1.5*IQR, Q3+1.5*IQR] as the boundary of the normal range to check and filter outliers.  

> NaN  
>> * After trying various filling methods such as "average", "median", "mode" and "KNN prediction", based on the performance applied to the model, the "average" filling method was finally adopted.  
<br>

## Model training (LightGBM)
#### The overall training process can be divided into two parts as shown in the figure.
#### Part 1: Find the best hyperparameter  
> We randomly sample 80% of the processed training data and perform 2-fold cross-segmentation (using stratified sampling to ensure that the proportions of various samples in the training set and validation set are the same as the original data set), and then use "Bayesian Optimization" and "GridSearchCV" to search for the best hyperparameter combination.  
> <br>
> More specifically, we first use "Bayesian Optimization" to perform a wide range of hyperparameter search (it builds a probability model of the objective function, and each hyperparameter selection is based on the previous evaluation, so it will be faster and more effective than Grid Search).  
> <br>
> Then, we use "GridSearchCV" to search in a small range, and after obtaining the best parameters calculated by the optimizer, we then manually adjust the final parameter combination in small increments.  
<br>

#### Part 2: Model training and prediction  
> Finally, we perform 10-fold cross-segmentation for all the training data, each time 1 fold is taken as the verification set, and the remaining 9 folds are used as the training set.  
> <br>
> After completing the 10-fold training, we will get a total of 10 prediction models. We select the 4 models that perform best on the training set, and average their predicted values as the final submitted predicted values.  
<br>

#### The highest prediction accuracy of the LightGBM model is about 0.85  

![avatar](C:\Users\doggy\Desktop\履歷範本\tbrain-LightGBM.png)

