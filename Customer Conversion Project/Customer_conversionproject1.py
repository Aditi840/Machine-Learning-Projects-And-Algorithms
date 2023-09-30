# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 15:02:18 2023

@author: Aditi
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import seaborn as sns
from matplotlib import pyplot as plt
import os

#Load the dataset
data = pd.read_csv("C:/Users/Aditi/Downloads/Customer Conversion Prediction - Customer Conversion Prediction.csv")



print(data.head())
print(data.isna().sum())
print(data.dtypes)
print(data["job"].value_counts)
eduRatio = pd.DataFrame({'Job' : []})
for i in data["job"].unique():
    edu_counts = data[data["job"] == i]["education_qual"].value_counts()
    edu_ratio = (edu_counts.iloc[0] * 100 / edu_counts.sum())
    edu_ratio_df = pd.DataFrame([edu_ratio], columns=["education_ratio"])
    eduRatio = pd.concat([eduRatio, edu_ratio_df], ignore_index=True)
eduRatio["Job"] = data["job"].unique()
print(eduRatio)
data.loc[(data.job == "unknown") & (data.education_qual == "secondary"),"job"] = "services"
data.loc[(data.job == "unknown") & (data.education_qual == "primary"),"job"] = "housemaid"
data.loc[(data.job == "unknown") & (data.education_qual == "tertiary"),"job"] = "management"
data.loc[(data.job == "unknown"),"job"] = "blue-collar"
print(data["job"].value_counts)

print(data["marital"].value_counts())

data.loc[(data.education_qual == "unknown") & (data.job == "admin."),"education_qual"] = "secondary"
data.loc[(data.education_qual == "unknown") & (data.job == "management"),"education_qual"] = "secondary"
data.loc[(data.education_qual == "unknown") & (data.job == "services"),"education_qual"] = "tertiary"
data.loc[(data.education_qual == "unknown") & (data.job == "technician."),"education_qual"] = "secondary"
data.loc[(data.education_qual == "unknown") & (data.job == "retired"),"education_qual"] = "secondary"
data.loc[(data.education_qual == "unknown") & (data.job == "blue-collar"),"education_qual"] = "secondary"
data.loc[(data.education_qual == "unknown") & (data.job == "housemaid."),"education_qual"] = "primary"
data.loc[(data.education_qual == "unknown") & (data.job == "self-employed"),"education_qual"] = "tertiary"
data.loc[(data.education_qual == "unknown") & (data.job == "student"),"education_qual"] = "secondary"
data.loc[(data.education_qual == "unknown") & (data.job == "entrepreneur"),"education_qual"] = "tertiary"
data.loc[(data.education_qual == "unknown") & (data.job == "unemployed"),"education_qual"] = "secondary"
#REST CAN BE SECONDARY
data.loc[(data.education_qual == "unknown"),"education_qual"] = "secondary"
print(data["education_qual"].value_counts())    
print(data.columns)
print(data["call_type"].value_counts())
data["call_type"].replace(["unknown"],data["call_type"].mode(),inplace=True)
# I replace unknown contact values with mode value
print(data["call_type"].value_counts())
print(data["prev_outcome"].value_counts())
#No need for ID column for training. Also dataset has pday column(number of days that passed by after last call).No need for "day" , "month" column for training
data.drop(columns = ["day","mon"],inplace=True)

#one hot encoding of job column
ohe = OneHotEncoder()
job_encoded = pd.DataFrame(ohe.fit_transform(data[["job"]]).toarray(), columns=["job_" + i for i in np.sort(data["job"].unique())])
data = pd.concat([data, job_encoded], axis=1)
data.drop(columns = ["job"],inplace=True)
#Marital column has 3 values lets apply OneHotEncoding again.
marital_encoded = pd.DataFrame(ohe.fit_transform(data[["marital"]]).toarray(), columns=["marital_" + i for i in np.sort(data["marital"].unique())])
data = pd.concat([data, marital_encoded], axis=1)
data.drop(columns = ["marital"],inplace = True)

print(data.head())
#Now we can label encode educational column.Beacause its ordinal data.
data.loc[(data.education_qual == "tertiary"),"educational_qual"] = 2
data.loc[(data.education_qual == "secondary"),"educational_qual"] = 1
data.loc[(data.education_qual == "primary"),"educational_qual"] = 0

#contact column label encoding
data.loc[(data.call_type == "telephone"),"call_type"] = 1 # 0 means cellular 1 means telephone
data.loc[(data.call_type == "cellular"),"call_type"] = 0
data.drop(columns = ["dur"],inplace=True)

prev_outcome_encoded = pd.DataFrame(ohe.fit_transform(data[["prev_outcome"]]).toarray(), columns=["prev_outcome_" + i for i in np.sort(data["prev_outcome"].unique())])
data = pd.concat([data, prev_outcome_encoded], axis=1)
data.drop(columns = ["prev_outcome"],inplace = True)
print(data.info())


#Before training we should transform object dtypes to int because some classifiers won't work with object dtype
#We are dropping the target y variable from data

data.call_type = data.call_type.astype(int)
data['education_qual'] = data['education_qual'].replace('primary', 0)
data['education_qual'] = data['education_qual'].replace('secondary', 1)
data['education_qual'] = data['education_qual'].replace('tertiary', 2)
data['education_qual'] = data['education_qual'].astype(int)
#lets split the data
X = data.drop("y", axis=1)
y = data["y"]
data.drop(columns = ["y"], inplace=True)
print(data.info())

plt.figure(figsize = (20,10))
sns.heatmap(data.corr(), annot = True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

#Logistic Regression
lr = LogisticRegression()
lr.fit(X_train,y_train)
y_predlr = lr.predict(X_test)
cmlr = confusion_matrix(y_test, y_predlr)
acclr = accuracy_score(y_test, y_predlr)
y_pred_lr = lr.predict_proba(X_test)[:,1]
lr_auc = roc_auc_score(y_test, y_pred_lr)
print("AUROC score for logistic regression without smote: ", lr_auc)
print(cmlr)
print(acclr)

#Data Augumentation
sm = SMOTE()
X_sm, y_sm = sm.fit_resample(X, y)

#Training after smote
X_train_sm, X_test_sm, y_train_sm, y_test_sm = train_test_split(X_sm, y_sm, test_size=0.25, random_state=10)
lr2 = LogisticRegression()
lr2.fit(X_train_sm, y_train_sm)
y_predlr2 = lr2.predict(X_test_sm)
y_pred_lr2 = lr.predict_proba(X_test_sm)[:,1]
cmlr2 = confusion_matrix(y_test_sm, y_predlr2)
acclr2 = accuracy_score(y_test_sm, y_predlr2)
lr2_auc = roc_auc_score(y_test_sm, y_pred_lr2)
print("AUROC score for logistic regression: ", lr2_auc)
print(cmlr2)
print(acclr2)

#Support Vector Classifier
svc = SVC(probability=True)
svc.fit(X_train_sm, y_train_sm)
y_predsvc = svc.predict(X_test_sm)
y_pred_svc = svc.predict_proba(X_test_sm)[:,1] # predict probabilities of positive class
cmsvc = confusion_matrix(y_test_sm, y_predsvc)
accsvc = accuracy_score(y_test_sm, y_predsvc)
svc_auc = roc_auc_score(y_test_sm, y_pred_svc)
print("AUROC score for svc: ", svc_auc)
print(cmsvc)
print(accsvc)

#KNN
knn = KNeighborsClassifier()
knn.fit(X_train_sm, y_train_sm)
y_predknn = knn.predict(X_test_sm)
y_pred_knn = knn.predict_proba(X_test_sm)[:,1] # predict probabilities of positive class
cmknn = confusion_matrix(y_test_sm, y_predknn)
accknn = accuracy_score(y_test_sm, y_predknn)
knn_auc = roc_auc_score(y_test_sm, y_pred_knn)
print("AUROC score for knn: ", knn_auc)
print(cmknn)
print(accknn)

#RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train_sm, y_train_sm)
y_predrf = rf.predict(X_test_sm)
y_pred_rf = rf.predict_proba(X_test_sm)[:,1] # predict probabilities of positive class
cmrf = confusion_matrix(y_test_sm, y_predrf)
accrf = accuracy_score(y_test_sm, y_predrf)
print(cmrf)
print(accrf)
rf_auc = roc_auc_score(y_test_sm, y_pred_rf)
print("AUROC for random forest: ", rf_auc)