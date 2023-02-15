# Credit_Risk_Analysis
Use Python to build and evaluate several machine learning models to predict credit risk.

## Overview

Credit risk is an inherently unbalanced classification problem, as good loans easily outnumber risky loans. For this project, we will employ different techniques to train and evaluate models with unbalanced classes using imbalanced-learn and scikit-learn libraries to build and evaluate models using resampling. Using the credit card credit dataset from LendingClub, a peer-to-peer lending services company we are going to do the following strategies to help us predict credit risk: 

- Oversample the data using the RandomOverSampler and SMOTE algorithms
- Undersample the data using the ClusterCentroids algorithm
- Use a combinatorial approach of over- and undersampling using the SMOTEENN algorithm
- Compare two additional machine learning models that reduce bias, BalancedRandomForestClassifier and EasyEnsembleClassifier

Using each of the techniques, we will then evaluate the performance of each model and make recommendations on whether any of these models can be used to predict credit risk.

## Results
### MODEL 1: Naive Random Oversampling
<p align="center">
  <img src="Images/Naive_Random_Oversampling.png">
  </p>
<p align = "center">
Fig.1 - Naive Random Oversampling Model Output
</p>

- The balanced accuracy score of .64 this tells us that the overall accuracy of this algorithm on this data set is 64%.
- The precision for high risk applicants is 0.01 which tells us this model had very few true positive results for high risk applicants.
- The precision for low risk applicants is 1 which tells us that this model had 100% accuracy in determining true positives for low risk applicants. (Most likely due to the high number of low risk applicants used from the data).
- The recall for high risk applicants in .65 which means 65% of the 101 high risk applicants where found to be high risk.
- The recall for low risk applicants in .63 which means 63% of the 17401 low risk applicants where found to be low risk.


### MODEL 2: SMOTE Oversampling 
<p align="center">
  <img src="Images/SMOTE_Oversampling.png">
  </p>
<p align = "center">
Fig.2 - SMOTE Oversampling Model Output
</p>

### MODEL 3: Undersampling 
<p align="center">
  <img src="Images/Undersampling.png">
  </p>
<p align = "center">
Fig.3 - Undersampling Model Output
</p>

### MODEL 4: Combination (Over and Under) Sampling 
<p align="center">
  <img src="Images/Combination_Sampling_SMOTEENN.png">
  </p>
<p align = "center">
Fig.4 - Combination Sampling (SMOTEENN) Model Output
</p>

### MODEL 5: Balanced Random Forest Classifier 
<p align="center">
  <img src="Images/Balanced_Random_Forest_Classifier.png">
  </p>
<p align = "center">
Fig.5 - Balanced Random Forest Classifer Model Output
</p>

### MODEL 6: Easy Ensemble AdaBoost Classifier 
<p align="center">
  <img src="Images/Easy_Ensemble_AdaBoost_Classifier.png">
  </p>
<p align = "center">
Fig.6 - Easy Ensemble AdaBoost Classifer Model Output
</p>

## Summary
