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
#lr_model = LogisticRegression(class_weight = 'balanced', 
  #                            max_iter = 1000,
 #                             penalty = 'none',
   #                           solver = 'saga', 
#    #                         random_state = 0)
#lr_model.fit(features, outcome)
#pred_probs = lr_model.predict_proba(features)

# calibrate the probabiities 
#platts_scaling = CalibratedClassifierCV(lr_model, cv = 2, method = "isotonic")
#platts_scaling.fit(features, outcome)
#calibrated_probs = platts_scaling.predict_proba(features)[:,1]

# %%
import pickle 

filename = 'model_output/final_logistic_isotonic_model.sav'

loaded_model = pickle.load(open(filename, 'rb'))

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

age = st.slider("Age", min_value = 21, max_value = 100, value = 80)
sex = st.selectbox("Sex", options = ["Male", 'Female'])


"""
### Tumor characteristics:
"""
tumor_size = st.slider("Tumor size", min_value = 0.1, max_value = 20.0, value = 1.0)

tumor_site = st.selectbox("Tumor location",
                          options = ["Head or neck", "Trunk",  "Extremity", "Other"])


lymph_vasc_invasion = st.selectbox("Presence of lymphnode vascular invasion",
                                   options = ["Yes", 'No'])
growth_pattern = st.selectbox("Infiltrative growth pattern",
                                   options = ["Yes", 'No'])

tumor_lymphocytes = st.selectbox("Presence of tumor lymphocytes", 
                             options = ["Yes", 'No'])

# %% PROCESS INPUT DATA 


# combine into dataframe
features_input = {"AGE": [age],
                  "tumor_size": [tumor_size],
                  "SEX": [sex],
                  "tumor_site": [tumor_site],
                  "growth_pattern" : [growth_pattern],
                  "lymph_vasc_invasion": [lymph_vasc_invasion],
                  "tumor_lymphocytes": [tumor_lymphocytes]
                  }

features_input = pd.DataFrame(features_input)


mean_age = 73.407
mean_tumor_size = 2.011
# mean-center the continuous vvariables
features_input['AGE'] = features_input.AGE - mean_age
features_input['tumor_size'] = features_input.tumor_size - mean_tumor_size

# process the dummy codedd variables 
features_input['SEX'] = features_input.SEX == "Male"
features_input['tumor_lymphocytes'] = features_input.tumor_lymphocytes == "Yes"
features_input['lymph_vasc_invasion'] = features_input.lymph_vasc_invasion == "Yes"
features_input['growth_pattern'] = features_input.growth_pattern == "Yes"
features_input['tumor_site_trunk'] = features_input.tumor_site == "Trunk"
features_input['tumor_site_head_neck'] = features_input.tumor_site == "Head or neck"
features_input['tumor_site_other'] = features_input.tumor_site == "Other"

features_input = features_input.drop(columns = 'tumor_site')
# reference, don't code
#features_input['tumor_site_extremity'] = features_input.tumor_site == "Extremity"

# %%

key_thresh = 0.79
    

pred_prob_adjusted_array = loaded_model.predict_proba(features_input)
pred_prob_adjusted = round(pred_prob_adjusted_array[0,1],2)

# assign class
if pred_prob_adjusted < key_thresh :
    pred_class = "Positive"
    biopsy = "Yes"
else:
    pred_class = "Negative"
    biopsy = "No"
    
"""
## Result
""" 

# subtracting from 1, because we predicted no-metas as the positive class
st.write("Probability of Metastasis", round(1- pred_prob_adjusted,2))
st.write('Class Assignment:', pred_class)
st.write('Recommend for biopsy:', biopsy)


   
   