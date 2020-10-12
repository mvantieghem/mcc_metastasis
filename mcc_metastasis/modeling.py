#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:50:27 2020
Custom functions for ML modeling 
@author: michellevantieghem2
"""
import sklearn
import pandas as pd
import numpy  as np
import math
import warnings

import seaborn as sns
import matplotlib.pyplot as plt
sns.set(context='paper', style='whitegrid', rc={'figure.facecolor':'white'}, font_scale=1.2)

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import recall_score, precision_score
from sklearn.metrics import accuracy_score, f1_score, brier_score_loss
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import make_scorer, auc 

from sklearn.calibration import CalibratedClassifierCV
from sklearn.calibration import calibration_curve



def get_ordered_coefs(model, X):
    """ get coefficients from model fit, in orde
    

    Parameters
    ----------
    model : fitted model object
        from sklearn Logistic regression
    X : dataframe 
        containing featureset used for model fit

    Returns
    -------
    coefs_df : dataframe
        includes coefficients and featurenames, ordered by coefs
        to be used in feature importance plot

    """
    coefs = pd.DataFrame(model.coef_.T)
    features = pd.DataFrame(X.columns)
    coefs_df = pd.concat([features, coefs], axis=1)
    coefs_df.columns = ['features', 'coefs']
    # sort the coefficients by value 
    coefs_df = coefs_df.sort_values(by = 'coefs', 
                           ascending= False, ignore_index=True)
    return coefs_df


# custom func
def pretty_cm(confmat, filename):

     """
     this creates the matplotlib graph to make the confmat look nicer
     """
     fig, ax = plt.subplots(figsize=(6, 4))
     ax.matshow(confmat, cmap=plt.cm.Reds, alpha=0.3)
     for i in range(confmat.shape[0]):
         for j in range(confmat.shape[1]):
             ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center', fontsize = 20)
       
     ax.set_xticklabels(['']+['Mets', 'No Mets'])
     ax.set_yticklabels(['']+['Mets', 'No Mets'])
     plt.xlabel('Predicted Label', size = 20)
     plt.ylabel('True Label', fontsize = 20)
     plt.xticks(fontsize = 15)
     plt.yticks (fontsize = 15)
     ax.xaxis.set_label_position('top')
     plt.tight_layout()
     plt.savefig(filename)
     plt.show()
  
def get_cm_info (cm, y_test):
    """print out summary of confusion matrix interpretation"""
    
    true_negatives = cm[0,0]
    false_negatives = cm[1,0]
    true_positives = cm[1,1]
    false_positives = cm[0,1]
    total_cases = y_test.shape[0]
    print("TN  %0.2f" % true_negatives)
    print("TP %0.2f" %  true_positives)
    print("FN %0.2f" %  false_negatives)
    print("FP %0.2f" %  false_positives)

    true_negative_prop = true_negatives / total_cases
    true_positive_prop = true_positives/ total_cases 
    recall = true_positives / (false_negatives + true_positives) # true positive rate

    
    false_negative_rate = false_negatives / (true_positives + false_negatives)
    false_positive_rate = false_positives / (false_positives + true_negatives)

    print("True positives identified: %0.2f \n Proportion of sample correctly identified  as low risk, no biopsy necessary" % true_positive_prop)
    print("True positive rate / recall / sensitivity: %02.f \n Of the people who don't have metastasis, how many are correctly identifiied" % recall)
    print("False positive rate %0.2f \n Patients mis-classified as low risk, but need biopsy:" % false_positive_rate)

    
    print("True negatives identified: %0.2f \n Proportion of sample correctly identified as metastasis, need biopsy" % true_negative_prop)
    print("False negative rate: %0.2f \n Of those classified as metastasis, how many are reallly no metastasis" % false_negative_rate)


   # precision = round(true_positives / (false_positives + true_positives),2)
   # npp = round(true_negatives/ (true_negatives + false_negatives),2)
   # specificity =  round(true_negatives/  (true_negatives + false_positives), 2)
   # print ("Recall / Sensitivity: \n Of all people who actually have metastasis, how many are correctly recommended for biopsy? {}".format(round(recall, 2)))

   # print('\n Specificity/True negative rate: \n Of people who are actually low risk, how many are classified as low risk?', format(specificity))

    
    #print ("Precision / Positive Predictive Value: \n Of all people recommended for biopsy, how many actually have metastasis? {}".format(round(precision, 2)))
    #print('Negative Predictive Value: \n Of people classified as low risk,  how many are actually low risk? {}'.format(round(npp)))
  

# https://johaupt.github.io/scikit-learn/tutorial/python/data%20processing/ml%20pipeline/model%20interpretation/columnTransformer_feature_names.html

def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names



