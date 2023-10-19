#!/usr/bin/env python
# coding: utf-8

# In[22]:


import pandas as pd
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np


# In[23]:


#Loading data
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')


# In[24]:


train_data.head()


# In[25]:


test_data.head()


# In[26]:


#Setting passenger Id as index column
train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")


# In[27]:


train_data.info()


# # Data Cleaning

# In[28]:


#We want to fill in the missing age data instead of just dropping the missing age data rows. One way to do this is by filling in the mean age of all the passengers (imputation)
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        elif Pclass == 2:
            return 29
        else:
            return 24
    else:
        return Age


# In[29]:


train_data['Age'] = train_data[['Age','Pclass']].apply(impute_age,axis=1)


# In[30]:


train_data


# In[31]:


#Based on the heat map cabin is no longer needed
train_data.drop('Cabin',axis=1,inplace=True)


# In[32]:


train_data.describe()


# In[33]:


train_data["Survived"].value_counts()


# In[34]:


train_data["Pclass"].value_counts()


# In[35]:


train_data["Pclass"].value_counts()


# In[36]:


train_data["Sex"].value_counts()


# In[37]:


train_data["Embarked"].value_counts()


# In[38]:


train_data.head()


# # Question 1: What are the odds of survival ?

# ### To calculate the odds of surviving we can use the number of survivors over the number of deaths. By taking the 342 survivors over the 549 deaths we can see that for every 0.6221 people who survived, 1 person did not survive, or roughly 62.21% of the passengers did not survive.

# # Question 2: Create dummy variable for gender

# In[39]:


sex = pd.get_dummies(train_data['Sex'],drop_first=True)
embark = pd.get_dummies(train_data['Embarked'],drop_first=True)


# In[40]:


train_data.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)


# In[41]:


train_data = pd.concat([train_data,sex,embark,],axis=1)


# In[42]:


train_data


# # Question 3: . Using the logit model, dependent variable: Survived, independent variable: age and gender

# In[46]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train_data.drop('Survived',axis=1), 
                                                    train_data['Survived'], test_size=0.30, 
                                                    random_state=101)
                                   


# In[66]:


logmodel = LogisticRegression(max_iter=3000)
logmodel.fit(X_train,y_train)

predictions = logmodel.predict(X_test)


# In[67]:


print(classification_report(y_test,predictions))


# In[70]:


logmodel.coef_


# In[ ]:





# # Question 3a: Report accuracy using 5-fold cross-validation

# In[51]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier


# In[52]:


target = train_data['Survived']
clf = KNeighborsClassifier(n_neighbors = 13)
scoring = 'accuracy'
score = cross_val_score(clf, train_data, target, cv=k_fold, n_jobs=1, scoring=scoring)
print(score)


# # Question 3b: Report confusion matrix

# In[292]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# In[293]:


y_test_predicted=logmodel.predict(X_test)


# In[294]:


cm=confusion_matrix(y_test,y_test_predicted,labels=logmodel.classes_)
print(cm)


# In[295]:


cm_disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=['dead','alive'])
cm_disp.plot()


# # Question 3c: What is precision score? How do you interpret the score?

# In[296]:


print(classification_report(y_test,predictions))


# #### our precision score is .79 which is not to high meaning that the model could be fittend and trained a little better.

# # Question 3d: What is recall score? How do you interpret the score?

# In[302]:


print(classification_report(y_test,predictions))


# #### Our recall score is .77 which means more positive results couldve been returned if the model was trained better.

# # Question 3e: Report the precision/recall trade-off plot?

# In[333]:


from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score,precision_recall_curve,roc_curve,roc_auc_score,make_scorer

print('Recall score\n')
print(recall_score(y_test,predictions))
print('\nPrecision score: \n')
print(precision_score(y_test,predictions))
print('\nAccuracy score \n')
print(accuracy_score(y_test,predictions))


# In[334]:


precision,recall,thresholds = precision_recall_curve(y_test,predictions)

plt.figure()
plt.plot(precision,recall)
plt.title('Precision-Recall Curve \nPrecision and Recall at each value of decision boundary')
plt.xlabel('Precision')
_ = plt.ylabel('Recall')


# # Question 3f: Report the ROC curve?

# In[342]:


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=3000, random_state=42)
sgd_clf.fit(X_train,y_train)

y_scores = cross_val_predict(sgd_clf, X_train, y_train, cv=5, method="decision_function")
y_scores

fpr, tpr, thresholds = roc_curve(y_train, y_scores)

def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0,1],[0,1],'k--')
    plt.axis([0,1,0,1])
    plt.xlabel("fpr")
    plt.ylabel("tpr")
plot_roc_curve(fpr, tpr)
plt.show()


# # Question 3g: What is area under curve?

# In[346]:


from sklearn.metrics import average_precision_score
average_precision = average_precision_score(y_train, y_scores)
average_precision


# #### based on the average precision score the AUC is .51

# # Question 3h: Report the logistic model

# In[71]:


logmodel = LogisticRegression(max_iter=3000)
logmodel.fit(X_train,y_train)
predictions = logmodel.predict(X_test)


# In[72]:


logmodel.coef_


# In[ ]:




