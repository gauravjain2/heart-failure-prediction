# HEART FAILURE PREDICTION APP

<br>

Data Science Project to assess the likelihood of a death event by heart failure.
This can be used to help hospitals in assessing the severity of patients with cardiovascular diseases.

Link to Website: [Click Here](https://sites.google.com/view/heart-failure-project/)

---

**About the dataset**

Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.

Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

---

**EDA**

Exploratory Data Analysis of the dataset 

![image](https://user-images.githubusercontent.com/43726919/132322608-e8d4aa71-be71-462c-ba81-096043d57108.png)

![image](https://user-images.githubusercontent.com/43726919/132322668-d057a5ec-f4d3-4f36-b6ac-2526fdec4650.png)

---

**MODEL**

Metrics Used: F1 score with a higher recall score and considerable precision score as the evaluation metrics.
Model training:
Created various models like:
1) Logistic Classifier	
2) Decision Tree Classifier 
3) Random Forest Classification Model
4) LGBM 
5) XGB 


and other classification models and fitted the train data to each model. 

Considering the bias variance trade-off and the evaluation metrics, the best performing model was Random Forest Classifier with optimized parameters.

![image](https://user-images.githubusercontent.com/43726919/132326245-625669b2-893e-48b7-8fc3-16e0dc0f25d4.png)

---

### What input data is required?
The model takes 12 variables as input:
- Age
- Anaemia
- Creatinine Phosphokinase
- Diabetes
- Ejection Fraction
- High Blood Pressure
- Platelets
- Serum Creatinine
- Serum Sodium
- Sex
- Smoking
- Follow Up Time in Hospital

---

### How to upload data for prediction?
Go to [Website](https://heart-failure-prediction-app.herokuapp.com/).
There are 2 options of uploading data for prediction:

1. Manual Entry (via sliders and text field): suitable for single patient prediction.
2. CSV file (Upload .csv file in the format specified): suitable for multiple predictions at same time.


---

*Disclaimer: This model is based on Data Science Practices and nowhere is affiliated with any medical authority. Please consult with your doctor for consultations/medicines.*
