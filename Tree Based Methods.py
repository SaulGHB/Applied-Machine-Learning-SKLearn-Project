#!/usr/bin/env python
# coding: utf-8

# # Downloading Necessary Packages 

# In[286]:


import pandas as pd
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix,recall_score,precision_score,accuracy_score,precision_recall_curve,roc_curve,roc_auc_score,make_scorer
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_curve
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import average_precision_score


# In[287]:


# Loading the data
#Loading data
train_data=pd.read_csv('train.csv')
test_data=pd.read_csv('test.csv')


# In[288]:


#Setting passenger Id as index column
train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")


# # Data Cleaning 

# In[289]:


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


# In[290]:


train_data


# In[291]:


train_data.dtypes


# # Question 1: What are the odds ratio of surviving the shipwreck between male and female?

# #### First we calculate male and female Survivors and Non-Survivors:
# #### Male Survivors = (577/891)x342 = 220
# #### Female Survivors = (314/891)x342 = 122
# #### Female NonSurvivors = 314-122=192
# #### Male NonSurvivors = 577-220 = 357
# 
# #### Next we can just calculte the odds ratio for each gender
# #### Odds of male surviving = 220/347=.616
# #### Odds of female surviving = 122/192=.635
# 
# #### Then the odds ratio would be .635/.616=1.03 and .616/.635=.97. The odds ratio is approximately 1.03, which means that the odds of a female passenger surviving were about the same as a male passenger. 
# 
# 

# # Question 2: Create dummy variable for Pclass

# In[292]:


train_data.drop(['Embarked','Name','Ticket','Cabin','Fare','Parch','SibSp'],axis=1,inplace=True)


# In[293]:


for col in train_data.dtypes[train_data.dtypes=="object"].index:
    for_dummy=train_data.pop(col)
    train_data=pd.concat([train_data,pd.get_dummies(for_dummy,prefix=col)],axis=1)
train_data.head()


# In[294]:


pclass = pd.get_dummies(train_data['Pclass'],drop_first=True)
train_data = pd.concat([train_data,pclass],axis=1)
train_data


# In[295]:


train_data.drop(['Pclass'],axis=1,inplace=True)
train_data


# # Question 3: Using the Random Forest, dependent variable: Survived, independent variable: age, gender, and Pclass.

# In[296]:


labels=train_data.pop("Survived")


# In[297]:


X_train, X_test, y_train, y_test = train_test_split(train_data,labels,test_size=0.30, random_state=101)


# In[298]:


from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()


# In[299]:


rf.fit(X_train,y_train)


# # COULD NOT GET TREE TO RUN AFTER SPLIT.

# In[ ]:




