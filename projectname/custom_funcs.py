#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:50:27 2020

@author: michellevantieghem2
"""
import pandas as pd
import numpy  as np
from sklearn.metrics import confusion_matrix

##  custom functions
def N_table(df, var):
    table =  (
        df
        .groupby(var, dropna = False)
        .size()
        .to_frame('N')
        .reset_index()
    )
    return table

def get_proportions(data, var):
    counts = pd.DataFrame(np.unique(data, return_counts = True), index = [var, 'Counts']).T
    counts['Percent'] = (counts['Counts']/counts['Counts'].sum()).round(2)*100
    counts  = counts.set_index(var)
    return counts 

def get_conf_matrix(y_test, y_pred):
    cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
    cm = cm.rename(columns = {0:' Predicted No Mets', 1:'Predicted Mets'}, 
         index = {0:'True no Mets', 1:'True Mets'})
    return cm

def get_cv_score_table(scores):
    score_table = pd.DataFrame(scores.mean()).\
    drop(['fit_time', 'score_time']).\
    rename(columns = {0:'Mean Score'}).\
    round(3)
    return score_table


def get_coefs(model, X_test):
    coefs = pd.DataFrame(model.coef_.T)
    features = pd.DataFrame(X_test.columns)
    coefs_df = pd.concat([features, coefs], axis=1)
    coefs_df.columns = ['features', 'coefs']
    # sort the coefficients by value 
    coefs_df = coefs_df.sort_values(by = 'coefs', 
                           ascending= False, ignore_index=True)
    return coefs_df
