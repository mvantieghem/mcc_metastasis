#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 09:19:21 2020

@author: michellevantieghem2
"""

# %%

import sklearn
import scipy
import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from scipy.stats import zscore

#print('The sklearn version is {}.'.format(sklearn.__version__))
#print(scipy.__version__)

#@st.cache
#def loadData():

# %%
# OPEN DATA 
df = pd.read_csv("data/cleaned/NCDB_cleaned_all_cases.csv")

# select features to include  
df_selected = df [['AGE',  'lymph_vasc_invasion', 
                   'tumor_size',  'metastasis']]

#drop extra stuff
df_drop = df_selected.dropna(axis = 0)

# center scale! 
#df_drop['AGE'] = zscore(df_drop.AGE)
#df_drop['tumor_size_bins_cm'] = zscore(df_drop.tumor_size_bins_cm)

outcome = df_drop['metastasis']
features = df_drop.drop(columns =['metastasis'])


# %% 

# fit the model on all data that exists
lr_model = LogisticRegression(class_weight = 'balanced', 
                              max_iter = 1000,
                              penalty = 'none',
                              solver = 'saga', 
                             random_state = 0)
lr_model.fit(features, outcome)
pred_probs = lr_model.predict_proba(features)

# calibrate the probabiities 
platts_scaling = CalibratedClassifierCV(lr_model, cv = 2, method = "isotonic")
platts_scaling.fit(features, outcome)
calibrated_probs = platts_scaling.predict_proba(features)[:,1]

# %% 

"""
# Modeling Metastasis
Predicting risk for metastasis in Merkel Cell Carcinoma
"""
 
 #  return score, report, lr_model
"""
## Input data
"""

#def accept_user_data():
"""
### Patient demographics:
"""

age = st.slider("Select Age", min_value = 21, max_value = 100, value = 80)
        
"""
### Tumor characteristics:
"""

lymph_vasc_invasion = st.radio("Presence of lymph vascular invasion",
                                   options = [0,1])
#tumor_size = st.select_box("Tumor size (cm)", 
 #                          "<1", "1-2", "2-3", "3-4", "4-5", "5+")
tumor_size = st.slider("Tumor size", min_value = 1, max_value = 6, value = 1)

features_input = {"AGE": [age], 
                "lymph_vasc_invasion": [lymph_vasc_invasion], 
                "tumor_size_bins_cm": [tumor_size]}
features_input = pd.DataFrame(features_input)
 
# %%
    
#user_prediction_data = accept_user_data()
pred_prob_array = lr_model.predict_proba(features_input)
pred_prob = round(pred_prob_array[0,1],2)


pred_prob_adjusted_array = platts_scaling.predict_proba(features_input)
pred_prob_adjusted = round(pred_prob_adjusted_array[0,1],2)

# assign class
if pred_prob_adjusted > 0.2 :
    pred_class = "Positive"
    biopsy = "Yes"
else:
    pred_class = "Negative"
    biopsy = "No"
    
"""
## Result
""" 

#st.write("Raw probability of Metastasis", pred_prob)
st.write("Probability of Metastasis", pred_prob_adjusted)
st.write('Class Assignment:', pred_class)
st.write('Recommend for biopsy:', biopsy)


   
   