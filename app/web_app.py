#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:19:21 2020

@author: michellevantieghem2
"""

# %%

import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#@st.cache
#def loadData():

# %%
df = pd.read_csv("../data/cleaned/NCDB_cleaned_N513.csv")
 # return df

#def preprocess(df):
outcome = df['metastasis']
features = df.drop(columns =['metastasis',
                            'regional_nodes_positive_bin',
                            'regional_nodes_ITC_bin',
                            'lymph_node_mets_bin'])
    
 # one-hot encode categorical variables  
one_hot_tumor_site = pd.get_dummies(features['tumor_site'], prefix =  "tumor_site")
features2 = features.drop(columns = ['tumor_site'])
features3 = features2.join(one_hot_tumor_site)
    
# split into train-test split
X_train, X_test, y_train, y_test = train_test_split(features3, outcome,
                                                   random_state = 0)

# %% 
#def classification_model(X_train, X_test, y_train, y_test):
# fit the model on all  of  training data 
lr_model = LogisticRegression(C = 0.01)
lr_model.fit(X_train, y_train)
y_pred = lr_model.predict(X_test)
#score = lr_model.score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# %% 

"""
# Predicting Probability of Metastasis
in Merkel Cell Carcinoma
"""
 
 #  return score, report, lr_model
"""
## Input data
"""

#def accept_user_data():
"""
### Patient demographics:
"""

age = st.slider("Select Age", min_value = 21, max_value = 100)
sex = st.selectbox("Sex", options = [1,2]) # options..
immuno_suppressed = st.radio("Does patient have immunosuppression?", 
                                 options = [ 0,1])
    
"""
### Tumor site:
"""
tumor_site_head_neck = st.checkbox("Head or neck")
tumor_site_trunk = st.checkbox("Trunk")
tumor_site_extremity = st.checkbox("Extremity")
tumor_site_other = st.checkbox("Other")
    
"""
### Tumor characteristics:
"""

lymph_vasc_invasion = st.radio("Presence of lymph vascular invasion",
                                   options = [0,1])
tumor_lymphocytes = st.radio("Presence of tumor-infiltrating lymphocytes", 
                                 options = [0,1])
tumor_depth = st.slider("Tumor dept (mm)", min_value = 0, max_value = 100)
tumor_size = st.slider("Tumor size (cm)", min_value = 0, max_value = 10, step = 1)
    
features = {"AGE": [age], 
                "SEX": [sex], 
                "tumor_site_head_neck": [tumor_site_head_neck],
                "tumor_site_trunk" : [tumor_site_trunk],
                "tumor_site_extremity" : [tumor_site_extremity],
                "tumor_site_other": [tumor_site_other],
                "lymph_vasc_invasion": [lymph_vasc_invasion], 
                "tumor_lymphocytes": [tumor_lymphocytes],
                "immuno_suppressed": [immuno_suppressed], 
                "tumor_depth": [tumor_depth], 
                "tumor_size": [tumor_size]}
features = pd.DataFrame(features)
 
# %%
    
#user_prediction_data = accept_user_data()
pred_prob_array = lr_model.predict_proba(features)
pred_prob = round(pred_prob_array[0,1],2)

"""
## Result
""" 

st.write("Estimated probability of Metastasis", pred_prob)


   
   