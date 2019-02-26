#!/usr/bin/env python
# coding: utf-8

# In[12]:


import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, cohen_kappa_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# In[13]:


TEST_RATIO = 0.2


# In[14]:


class LDAModel:
    def __init__(self):
        self.data = pd.DataFrame()
        self.target = pd.DataFrame()
        self.classifier = LDA()
        self.mae = 0
        self.mse = 0
        self.matrix = None
        
    def load_data(self, filepath):
        self.data = pd.read_csv(filepath)
        self.data.head()
        return self.data
    
    def load_target(self, filepath):
        self.target = pd.read_csv(filepath)
        self.target.head()
        return self.target
        
    def train(self, data=None, target=None):
        # uses own dataset and labels if none supplied
        if data is None:
            data = self.data
        if target is None:
            target = self.target

        # splits data into training and testing datasets
        train_data, train_target, test_data, test_target = train_test_split(data, target, test_size=TEST_RATIO)
        
        #train LDA model
        clf = LDA()
        clf.fit(train_data, train_target)
        
        # predicts data on test dataset
        test_true, test_pred = test_target, clf.predict(test_data)
        
        # gets evaluation metrics from predictions
        mse = mean_squared_error(test_true, test_pred)
        mae = mean_absolute_error(test_true, test_pred)
        matrix = confusion_matrix(test_true, test_pred)

        # prints out metrics
        print("\nMean Square Error:", mse)
        print("Mean Absolute Error:", mae)
        print("Confusion Matrix:\n", matrix)
        
        #returns trained LDA model
        self.classifier = clf
        return clf

    def disp_confusion_matrix(self):
        print(self.matrix)


# In[ ]:




