# CDC-Diabetes-Health-Indicator

CDC Diabetes Health Indicator

The Diabetes Health Indicators Dataset contains healthcare statistics and lifestyle survey information about people in general along with their diagnosis of diabetes. The 35 features consist of some demographics, lab test results, and answers to survey questions for each patient. The target variable for classification is whether a patient has a value of (1) diabetes and/or pre-diabetic, or a value of (0) healthy.

Dataset is from UCI Dataset: Link: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

<img width="1286" alt="Screenshot 2024-09-07 at 10 22 30 AM" src="https://github.com/user-attachments/assets/fa99d314-9d9c-4328-b95c-d0adb4fb2ab6">

Other important documents for reference are : 
https://www.cdc.gov/brfss/annual_data/2014/pdf/CODEBOOK14_LLCP.pdf
https://www.cdc.gov/mmwr/volumes/66/wr/mm6643a2.htm

## Overview:
In this capstone project, my goal is to understand the relationship between the lifstyle of the people in the US to diabetes. The dataset from CDC provides enough information about the lifestyle of both male and female participants. With this data, I want to identify patterns that can potentially increase the chances of diabetes. Ths capstone project will explore the classification models namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines. In the final submission, I will use hyperparametes to tune the model to determine patterns to identify patients with diabetes.

## Understanding the Data

The dataset comes with 21 attributes. There are no null values so all rows have information that can be utilized for data analysis and model evaluation.
There are 253,680 records for us to analyze. All the attributes are numerical. Some of the data has been bucketed into categories and assigned a numerical value. The attributes that fall into these categories of bucketing are : Age, Education, Income, and General Health (GenHlth). The attribute related to gender (Sex) identifies the participant as Male vs. Female. All other attributes are binary attributes that indicate a yes or no.

The dataset is broken into the following items:

#### Patient Information
- Age - 13-level age category (_AGEG5YR see codebook)
- Sex - Male or Female
- Education - Education level (EDUCA see codebook) scale 1-6
- Income - Income scale (INCOME2 see codebook) scale 1-8

#### Patient Health Information
- GenHlth              :  5 unique values. They are : [1, 2, 3, 4, 5]  Would you say that in general your health is: scale 1-5
- HighBP               :  2 unique values. They are : [0, 1]
- HighChol             :  2 unique values. They are : [0, 1]
- CholCheck            :  2 unique values. They are : [0, 1]
- Smoker               :  2 unique values. They are : [0, 1]
- Stroke               :  2 unique values. They are : [0, 1]
- HeartDiseaseorAttack :  2 unique values. They are : [0, 1]
- PhysActivity         :  2 unique values. They are : [0, 1]
- AnyHealthcare        :  2 unique values. They are : [0, 1]
- NoDocbcCost          :  2 unique values. They are : [0, 1]
- DiffWalk             :  2 unique values. They are : [0, 1]

- Fruits               :  2 unique values. They are : [0, 1]
- Veggies              :  2 unique values. They are : [0, 1]
- HvyAlcoholConsump    :  2 unique values. They are : [0, 1]

- MentHlth             : 31 unique values. 
- PhysHlth             : 31 unique values. 
- BMI                  : 84 unique values. 

#### Unique values for Patient Info
- Sex                  :  2 unique values. They are : [0, 1]
- Age                  : 13 unique values. They are : [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
- Education            :  6 unique values. They are : [1, 2, 3, 4, 5, 6]
- Income               :  8 unique values. They are : [1, 2, 3, 4, 5, 6, 7, 8]

#### Target Variable
- Diabetes_binary - has the patient been diagnosed as Diabetic or Not (0 or 1)

#### Missing Data needing special attention
- There are no missing data in any of the attributes

## Exploratory Data Analysis

There are a total of 14 attributes with only two values. 

![Patient Data - Binary Attributes](https://github.com/user-attachments/assets/c6b685a9-afcf-4964-a963-f408b29bdd4c)

####  Breaking down the data by Males and Felames, we get:

![Patient Information for Binary Attributes](https://github.com/user-attachments/assets/22a18cae-317f-46e4-9b26-c586efcfe358)

#### Pearson's Correlation of all attributes

I ran the Pearson Correlation. Below is the result of the correlation between all the numerical variables.

![Pearsons Correlation - Patient Data](https://github.com/user-attachments/assets/58705fa8-c1de-40ca-ba62-086ab24ba933)

## Principal Component Analysis

<img width="931" alt="Screenshot 2024-09-15 at 2 10 27 AM" src="https://github.com/user-attachments/assets/debff7a5-f817-4145-b976-53c508f7219e">

<img width="931" alt="Screenshot 2024-09-15 at 2 10 16 AM" src="https://github.com/user-attachments/assets/baedd780-2ddc-4bdd-b9b2-66457774ee2d">

## Model Evaluation

Using the refined dataset, I split this into 80% for training and 20% for testing.
To create standardized results for each of the model, I created a series of functions and called these functions for each of the models.

**Functions created:**
- **Print Performance:** This will print the performance results of each model. It will print the accuracy, recall, precision, f1 scores.
- **Print Confusion Matrix:** This will print the confusion matrix and the associated values of True Positive, True Negative, False Positive, and False Negative
- **Print ROC-AUC Scores:** This will plot the ROC-AUC Curve and print the ROC-AUC score.
- **Evaluate Function:** This will use either the default setting or the hyperparameter to call the model. Perform the model fit, predict, and calculate and print the processing time, performance, confusion matrix, and the ROC-AUC curve and scores.

### Model Comparison

I created a baseline of the model using `Dummy Classifer` and then evaluated the following models without any hyperparameter tuning.

The Confusion Matrix for the Dummy Classifier (as expected) is shown below.

<img width="700" alt="Screenshot 2024-09-15 at 3 02 10 AM" src="https://github.com/user-attachments/assets/98e3117a-4263-4cca-90c8-98014fc45967">

### Initial Model Comparison : Without Hyperparameter Tuning (using Default Settings of each model)

- **Dummy Classifier**
- **Logistic Regression**
- **Decision Tree Classifier**
- **K Nearest Neighbor Classifier**
- **Support Vector Machines**

Based on the analysis of the refined dataset, the results from these models were as folows:

#### Results from Model Evaluation using Default Settings for each Model

<img width="1390" alt="image" src="https://github.com/user-attachments/assets/32687bc7-d847-4171-9d1b-f64c7d7dd710">

#### Confusion Matrix using Default Settings for each Model - for Males

The associated Confusion Matrix for these models (excluded Dummy Classifier) are as shown below. 

![Confusion Matrix Comparison for 3 Models-Males](https://github.com/user-attachments/assets/3a198cfa-709b-4b7f-970e-f6a84cc68e34)

#### Confusion Matrix using Default Settings for each Model - for Females

The associated Confusion Matrix for these models (excluded Dummy Classifier) are as shown below. 

![Confusion Matrix Comparison for 3 Models-females](https://github.com/user-attachments/assets/a88bf9d4-27a5-4e72-8dc4-aed45d066205)

#### ROC AUC Curve using Default Settings for each Model - for Males

The associated ROC AUC Curve for each of these models (excluding Dummy Classifier) are as shown below.

![Optimized ROC-AUC Curve Comparison for 3 Models - Males](https://github.com/user-attachments/assets/8e702c7a-2987-4bf9-948a-ac0a042d6bd6)

#### ROC AUC Curve using Default Settings for each Model - for Females

The associated ROC AUC Curve for each of these models (excluding Dummy Classifier) are as shown below.

![Optimized ROC-AUC Curve Comparison for 3 Models - Females](https://github.com/user-attachments/assets/8a405f2b-9173-4a18-b8e5-2aadca269652)

#### Summary Scores for Males

<img width="1237" alt="Screenshot 2024-09-15 at 3 08 03 AM" src="https://github.com/user-attachments/assets/2d591f35-7b39-4ab9-ae0a-71f7e9293b33">

#### Summary Scores for Females

<img width="1237" alt="Screenshot 2024-09-15 at 3 09 11 AM" src="https://github.com/user-attachments/assets/9176ef92-6e2c-4a06-ab7e-73edb4e2f30b">

#### Observation:
- Based on the results shown above, we can see that Logistic Regression has a very good accuracy score of 0.87  
- Decision Tree Classifer has a lower test accuracy score of 0.81 but the training scores is 1.00. The recall score is much higher for Decision Tree  
- Looking at the performance, K Nearest Neighbor has the best training time while maintaining a competitive accuracy score. However, the test timing is much higher
- Overall, I would recommend Logistic Regresion as the choice of model if we were to scale the test to a bigger dataset as the accuracy score of 0.87 and the ROC-AUC curve is 0.83 is comparatively much higher than all others.

#### Opportunity:
- Improve the Recall Score as they are ranging from 0.30 (SVM) to 0.52 (Decision Tree)

### Improved Model Comparison : With Hyperparameter Tuning

#### Hyperparameter Selected:

- **LogisticRegression:**

```
    'Logistic Regression': {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs', 'saga']
```

- **Decision Tree Classifier:**

```
    'Decision Tree Classifier': {
        'classifier__criterion': ['gini', 'entropy'],
        'classifier__max_depth': [None, 10, 20, 30, 40, 50],
        'classifier__min_samples_split': [2, 5, 10]
```
- **K Nearest Neighbor Classifier:**

```
    'K Nearest Neighbor Classifier': {
        'classifier__n_neighbors': [3, 5, 7, 9, 11],
        'classifier__weights': ['uniform', 'distance'],
        'classifier__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
```
- **Support Vector Machines:**

Due to processing time challenges, I excluded processing SVC.

With the further refined dataset, below are the results for the four models:

#### Results from Model Evaluation using Hyperparameter Tuning for each Model

#### For Males:

#### Confusion Matrix using Hyperparameter Tuning for each Model

The associated Confusion Matrix for these models are as shown below.

![Optimized Confusion Matrix Comparison for 3 Models - Males](https://github.com/user-attachments/assets/6b526c54-7f41-4ec2-95e6-3364f9fbfe53)

#### ROC AUC Curve using Hyperparameter Tuning for each Model

The associated ROC AUC Curve for each of these models are as shown below.

![Optimized ROC-AUC Curve Comparison for 3 Models - Males](https://github.com/user-attachments/assets/b80485fc-990b-4c1f-897e-b88340a34da0)

#### For Females:

#### Confusion Matrix using Hyperparameter Tuning for each Model

The associated Confusion Matrix for these models are as shown below.

![Optimized Confusion Matrix Comparison for 3 Models - Females](https://github.com/user-attachments/assets/f9f77073-faa2-47cd-b8d7-5a8b033d790e)

#### ROC AUC Curve using Hyperparameter Tuning for each Model

The associated ROC AUC Curve for each of these models are as shown below.

![Optimized ROC-AUC Curve Comparison for 3 Models - Females](https://github.com/user-attachments/assets/af3ee291-cecb-47cd-80de-ab5a0accd19e)

#### Observation:
Adding hyperparameters, I was able to see a much better result for all 3 models. 
- The test accuracy ratio improved from **0.81** to **0.87** with the ROC-AOC score pushing from **0.7x** to **0.8x**.
- However, as we can see, the hyperparameters comes with a cost.
- The time to process this takes longer with kNN Classifier taking more than **13 minutes** to process compared to earlier process less than **1 seconds**.

## Recommendation:
Use Logistic Regression with hyperparameters as the precision is higher, recall is higher, and ROC AUC score is also higher. The overall time it takes to process Logistic Regression is much lower than all others making it the best option among the 3 models

## Results from Ensemble Techniques:
<img width="811" alt="Screenshot 2024-10-08 at 10 15 42 AM" src="https://github.com/user-attachments/assets/a5ed1882-77cc-40bd-83ce-b416dfea588b">

## Conclusion: 
Looking at the results from the Ensemble techniques, we see eXtreme Graident Boosting (XGB) Classifier providing the best results with a higher Precision and low Recall and high Accuracy.

Also, the processing time for XGBoost was much lower than the other ones. While Random Forest did give a good score, it may have done a lot more over fitting and my recommendation is to ignore that model.

Convert the XGBoost Classifier solution as an output and work on ML Operations to implement this as a production ready solution.

## Future Questions:
A few questions we can ask about the dataset are:
1. Can we break down the analysis by age , education level, and gender. Would that provide a better result
2. Can we identify specific features by using feature importance to determine the critical features that help determine whether a patient is diabetic or not

## Usage
This model can be used to learn more about the patients and determine the relationship between their lifestyle and diabetes.

### Requirements:
- **Python 3.x**
- **pandas, numpy, scikit-learn, matplotlib, seaborn (Python libraries)**
- **scikit-learn Models: DummyClassifier, LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier, SVC (for Support Vector Machines)**

### Running the Project:
- Model is ready for usage. The next step is to convert this into an API or into a .py file or create a wrapper to call the iPython Jupyter Notebook.
- Alternate would be to deploy this into a Deployment Platform like Azure, AWS, or Google Cloud.
- Converting this to a package and use tools like Papermill  (https://github.com/nteract/papermill) to parameterize notebooks and feed different inputs through them

**Example** of how Netflix uses Papermill to deploy code into production:

You can read all about this here: https://netflixtechblog.com/notebook-innovation-591ee3221233

![image](https://github.com/user-attachments/assets/f0429c50-2c4c-4aca-8a8d-80c462c12bd4)


## Clone the Repository:

You can clone my project from this repository

    https://github.com/FerndzJoe/CDC-Diabetes-Health-Indicator

My Jupyter Notebook can be directly accessed using this:

    https://github.com/FerndzJoe/CDC-Diabetes-Health-Indicator/blob/main/Capstone%20Project%20-%20CDC%20Diabetes%20Health%20Indicator.ipynb

## Repository Structure  

- `Capstone Project - CDC Diabetes Health Indicator.ipynb`: Contains the Jupyter Notebook with detailed code including comments and analysis.

  (https://github.com/FerndzJoe/CDC-Diabetes-Health-Indicator/blob/main/Capstone%20Project%20-%20CDC%20Diabetes%20Health%20Indicator.ipynb)

- `README.md`: Summary of findings and link to notebook
