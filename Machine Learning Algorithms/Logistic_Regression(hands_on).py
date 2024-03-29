# -*- coding: utf-8 -*-
"""Logistic Regression(Hands On).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1PsDqSwXaRpKAXl5jjGgGXX4-n4zyMKD5
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

candidates = {'gmat': [780,750,690,710,680,730,690,720,740,690,610,690,710,680,770,610,580,650,540,590,620,600,550,550,570,670,660,580,650,660,640,620,660,660,680,650,670,580,590,690],
              'gpa': [4,3.9,3.3,3.7,3.9,3.7,2.3,3.3,3.3,1.7,2.7,3.7,3.7,3.3,3.3,3,2.7,3.7,2.7,2.3,3.3,2,2.3,2.7,3,3.3,3.7,2.3,3.7,3.3,3,2.7,4,3.3,3.3,2.3,2.7,3.3,1.7,3.7],
              'work_experience': [3,4,3,5,4,6,1,4,5,1,3,5,6,4,3,1,4,6,2,3,2,1,4,1,2,6,4,2,6,5,1,2,4,6,5,1,2,1,4,5],
              'admitted': [1,1,0,1,0,1,0,1,1,0,0,1,1,0,1,0,0,1,0,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,0,0,0,0,1]
              }
df = pd.DataFrame(candidates, columns = ['gmat', 'gpa', 'work_experience', 'admitted'])
print(df)

df.isnull().sum()

df.describe()

df.dtypes

df = df.drop_duplicates()

"""**EDA**"""

plt.figure(figsize=(10,8))
sns.scatterplot(x='gmat', y='work_experience', hue = 'admitted', data=df, s=200)
plt.title("Educational Qualification Data", y=1.015, fontsize=20)
plt.xlabel("GMAT", labelpad=13)
plt.ylabel("Work Experience", labelpad=13)
ax = plt.gca()

"""# 1) High GMAT and high Work experience  = > admitted
2) there is a lienar decision boundary =? good to go with LR model
"""

plt.figure(figsize=(10,8))
sns.scatterplot(x='gmat', y='gpa', hue='admitted', data=df, s=200)
plt.title("Educational Qualificational Data", y=1.015, fontsize=20)
plt.xlabel("GMAT", labelpad=13)
plt.ylabel("GPA", labelpad=13)
ax = plt.gca()

plt.figure(figsize=(10,8))
sns.scatterplot(x='gpa', y='work_experience', hue='admitted', data=df, s=200)
plt.title("Educational Qualification Data", y=1.015, fontsize=20)
plt.xlabel("GPA", labelpad=13)
plt.ylabel("Work Experience", labelpad=13)
ax = plt.gca()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for s in df.admitted.unique():
  ax.scatter(df.gpa[df.admitted==s],df.gmat[df.admitted==s],df['work_experience'][df.admitted==s], label=s)
ax.legend()

"""1) When we look at the 3d plot, all the points in the top right are blue and all points in bottom left are orange => there is a clear linear decision boundary

2) No transfomration required

3) People with more gpa, gmat, work experience are more likely to get admitted => In line with our GK

**Data Jar**
"""

X = df[['gmat', 'gpa', 'work_experience']].values
y = df['admitted'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred = logistic_regression.predict(X_test)

y_pred

logistic_regression.predict([[650,4.0,4]])

logistic_regression.predict_proba([[650,4.0,4]])

logistic_regression.coef_

logistic_regression.intercept_

"""**Evaluation Metrics**"""

pip install --upgrade scikit-learn

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, auc
logistic_regression.score(X_test, y_test) #accuracy
confusion_matrix(y_test, y_pred)
f1_score(y_test, y_pred)
# Calculate ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()