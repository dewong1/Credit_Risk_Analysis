# Credit_Risk_Analysis
Apply machine learning to solve real-world challenge: credit card risk 
(build skills in data preparation, statistical reasoning, and machine learning) 

## Overview of Project

**Machine Learning** (*supervised, unsupervised, & deep*) is the use of statistical algorithms to perform tasks such as learning from data patterns and making predictions. The basic procedure for implementing a *supervised learning* model are-- 1) Create a model, 2) Train the model, and then 3) Create predictions. The two main uses of supervised learning are: *regression* and *classification*. For this project, we worked with Jill (the lead data scientist) to employ different techniques to train and evaluate models with unbalanced classes. We used *imbalanced-learn* and *scikit-learn* (python machine learning libraries) to build and evaluate models using resampling. Using credit card credit dataset from LendingClub, a peer-to-peer lending services company, we oversampled the data using the *RandomOverSampler* and *SMOTE* algorithms, and undersampled the data using the *ClusterCentroids* algorithm. Then, we used a combinatorial approach of over-and undersampling using the *SMOTEEN* algorithm. Next, we compared two new machine learning models that reduce bias, *BalancedRandomForestClassifier* and *EasyEnsembleClassifier* to predict credit risk. At the end, we evaluated the performance of these models and made a written recommendation on whether they should be used to predict credit risk. 


## Results

### Six machine learning models (balanced accuracy, precision, and recall scores) 


### Naive Random Oversampling

<img width="800" alt="Naive Random Sampling" src="https://user-images.githubusercontent.com/107021231/194020347-1f594286-eaca-4968-a3e5-574478a5a732.png">

* Balanced accuracy: ~0.66
* Precision: High-Risk (low precision); Low-Risk (high precision) 
* Recall: High/Low-risk = 0.62/0.68


### SMOTE Oversampling

<img width="800" alt="SMOTE Oversampling" src="https://user-images.githubusercontent.com/107021231/194020362-a92ce95f-f662-43eb-99d7-49ab72a03414.png">

* Balanced accuracy: ~0.63
* Precision: High-Risk (low precision); Low-Risk (high precision) 
* Recall: High/Low-risk = 0.69/0.62




### Undersampling

<img width="800" alt="Undersampling" src="https://user-images.githubusercontent.com/107021231/194020389-f7e88be7-b58f-468c-911f-ca8f9950102d.png">

* Balanced accuracy: ~0.63
* Precision: High-Risk (low precision); Low-Risk (high precision) 
* Recall: High/Low-risk = 0.59/0.44



### Combination (Over & Under) Sampling 

<img width="800" alt="Combination (under   over) Sampling" src="https://user-images.githubusercontent.com/107021231/194020437-9671d7d3-36bd-4993-b18f-3d0bd7eb3fbc.png">

* Balanced accuracy: ~0.52
* Precision: High-Risk (low precision); Low-Risk (high precision) 
* Recall: High/Low-risk = 0.71/0.53



### Balanced Random Forest Classifier

<img width="800" alt="Balanced Random Forest Classifier" src="https://user-images.githubusercontent.com/107021231/194178726-2b30cfad-72a9-4c85-bb45-5301f487123a.png">


* Balanced accuracy: ~0.78
* Precision:  High-Risk (low precision); Low-Risk (high precision) 
* Recall: High/Low-risk = 0.67/0.91


### Easy Ensemble AdaBoost Classifier 

<img width="800" alt="Easy Ensemble AdaBoost Classifier" src="https://user-images.githubusercontent.com/107021231/194178942-fec1e085-5a34-4ab5-b2dc-669b5b034c66.png">


* Balanced accuracy:  ~0.92
* Precision:  High-Risk (low precision); Low-Risk (high precision) 
* Recall: High/Low-risk = 0.90/0.94


## Summary

When working with balanced accuracy (the highest compared accuracy between 0 and 1), the closest to 1 is the best machine learning model. In this case, for this credit card data set, the Easy Ensemble AdaBoost Classifier was the best model to choose with its 93% accuracy and a high recall score. Overall, the Ensemble AdaBoost Classifier was the best machine learning model to choose for futher credit card analysis due to its good balance between precision and sensitivity, as well as accuracy. 

