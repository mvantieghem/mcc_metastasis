#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:50:27 2020

@author: michellevantieghem2
"""
import pandas as pd
import numpy  as np
import sklearn 
import math
import warnings
import matplotlib.pyplot as plt

# metrics for classification
from sklearn.metrics import roc_curve, precision_recall_curve
from sklearn.metrics import auc, recall_score, precision_score,f1_score, brier_score_loss

# post-processing 
from sklearn.calibration import calibration_curve
from sklearn.calibration import CalibratedClassifierCV



#%% Data cleaning 


def binarize_var(variable, NA_vector):
    """ binarize continous variables 
    
    Parameters
    ----------
    
    variable: float variable
        raw variable from NCDB 
        
    NA_vector: array of values
        values that should be replaced with NaN
        based on coding provided in NCDB 
        
    Returns
    ---------
    variable_bin: float variable consisting of 1 and 0
        representing binary 0 = False, 1 = True
    
    """
    variable_bin = (
        variable
        .replace(NA_vector, np.nan)
        .apply(lambda x: x if math.isnan(x) else x> 0)
        .replace({True: 1, False: 0})
    )
    return variable_bin

def recode_tumor_site(var):
    """ recode primary site for tumor location on body
    
    Parameters
    ---------
    
    var: variable (float)
        raw data from NCDB on tumor site 
        
    Returns
    --------
    recode_var: varirable (categorical)
        collapsing across certain tumor locations to generate categories
        4 levels: head_neck, trunk, extremity, other
    """
    if (var == 'C440') | (var == 'C441') | (var == 'C442') | (var == 'C444'):
        recode_var = 'head_neck'
    elif (var == 'C445'):
        recode_var = 'trunk'
    elif (var == 'C446') | (var == 'C447'):
        recode_var = 'extremity'
    else:
        recode_var = 'other'
    return recode_var

# recode variable: primary size, in cm
def recode_tumor_size(var):
    """ recodes tumor size into 1 cm bins

    Parameters
    ----------
    var : variable (float)
        raw data from NCDB on tumor size 

    Returns
    -------
    recode_var : variable (float)
        values provided as exact values are converted into bins
        all data converted into cm

    """
    if (var == 0):
        recode_var = 0
    elif (var == 991) | (var < 100):
        recode_var = 1
    elif (var == 992) | (var < 200):
        recode_var = 2
    elif (var == 993) | (var < 300):
        recode_var = 3
    elif (var == 994)  | (var < 400):
        recode_var = 4
    elif (var == 995) | (var < 500):
        recode_var = 5
    elif(var == 996) | (var < 990):
        recode_var = 6 # but this really means 5 +
    else:
        recode_var = None
    return recode_var

def recode_outcome (df):
    """ Recode outcome variable of metastasis

    Parameters
    ----------
    df : NCDB data
        cleaned data, after key variables have been binarized

    Returns
    -------
    variable for positive or negative metastasis
        based on agreement from regional nodes and lymph node mets 

    """
    # if they have both variables... (if both are NOT na) 
    if (not math.isnan(df.regional_nodes_positive_bin)) & (not math.isnan(df.lymph_node_mets_bin)):
        # provide value if they both agree
        if (df.regional_nodes_positive_bin == 1) & (df.lymph_node_mets_bin == 1): 
            return 1
        elif (df.regional_nodes_positive_bin == 0) & (df.lymph_node_mets_bin == 0): 
            return 0
        # if they don't agree, exclude 
        else: 
            return np.nan 
    # if they only have one variable, use that
    elif (not math.isnan(df.regional_nodes_positive_bin)):
        return df.regional_nodes_positive_bin
    elif (not math.isnan(df.lymph_node_mets_bin)):
        return df.lymph_node_mets_bin

        
 
    
#%% Making tables 

def get_missingness_table(df):
    missing_table = (
        df.isna()
        .sum()
        .to_frame("Missing")
        .reset_index()
    )
    print("Missingness of features")
    return missing_table

def get_N_table(df, var):
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


#%% Modeling functions

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

# %% Useful functions that I didn't write

# custom func
def pretty_cm(confmat, filename):

     """
     this creates the matplotlib graph to make the confmat look nicer
     """
     fig, ax = plt.subplots(figsize=(6, 4))
     ax.matshow(confmat, cmap=plt.cm.Blues, alpha=0.3)
     for i in range(confmat.shape[0]):
         for j in range(confmat.shape[1]):
             ax.text(x=j, y=i, s=confmat[i, j], va='center', ha='center', fontsize = 30)
       
     ax.set_xticklabels(['']+['No Mets', 'Mets'])
     ax.set_yticklabels(['']+['No mets', 'Mets'])
     plt.xlabel('Predicted Label', size = 20)
     plt.ylabel('True Label', fontsize = 20)
     plt.xticks(fontsize = 20)
     plt.yticks (fontsize = 20)
     ax.xaxis.set_label_position('top')
     plt.tight_layout()
     plt.savefig(filename)
     plt.show()
    
    

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



# https://medium.com/analytics-vidhya/probability-calibration-essentials-with-code-6c446db74265
def plot_calibration_curve(est, name, fig_index, X_train, X_test, y_train, y_test):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    isotonic = CalibratedClassifierCV(est, cv=2, method='isotonic')

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(est, name),
                      (isotonic, name + ' + Isotonic')]:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        if hasattr(clf, "predict_proba"):
            prob_pos = clf.predict_proba(X_test)[:, 1]
        else:  # use decision function
            prob_pos = clf.decision_function(X_test)
            prob_pos = \
                (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())

        clf_score = brier_score_loss(y_test, prob_pos, pos_label=y_test.max())
        print("%s:" % name)
        print("\tBrier: %1.3f" % (clf_score))
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))

        fraction_of_positives, mean_predicted_value = \
            calibration_curve(y_test, prob_pos, n_bins=10)

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                 label="%s (%1.3f)" % (name, clf_score))

        ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                 histtype="step", lw=2)

    ax1.set_ylabel("Fraction of positives", size = 15)
    ax1.set_ylim([-0.05, 1.05])
    ax1.legend(loc="lower right")
    ax1.set_title('Calibration plots  (reliability curve)', size = 20)

    ax2.set_xlabel("Mean predicted value", size = 16)
    ax2.set_ylabel("Count", size = 15)
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()


# %% PLOTTING FUNCTIONS

# custom plot code 
def plot_predprobs(y_probs, key_thresh):
    plt.figure(figsize = (10, 8))
    plt.hist(y_probs, alpha = 0.5)
    plt.axvline(linewidth = 4, color = "red", linestyle = "dotted",
                x = key_thresh, label = "lowered threshold %0.2f" % key_thresh)
    plt.axvline(linewidth = 4, color = "black", linestyle = "dotted",
                x = 0.5, label = "default threshold 0.5")
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.xlim(0, 1)
    plt.xlabel("Predicted Probability of Metastasis", size = 15)
    plt.ylabel ("Frequency",  size = 15)
    plt.legend(loc="center right", ncol=1)
    plt.show()
    
def get_cm_info (cm, y_test):
    """print out summary of confusion matrix interpretation"""
    
    true_negatives = cm[0,0]
    false_negatives = cm[1,0]
    true_positives = cm[1,1]
    false_positives = cm[0,1]
    total_cases = y_test.shape[0]

    true_negative_prop = true_negatives / total_cases
    true_positive_prop = true_positives/ total_cases 
    print("True negatives identified: Proportion of sample correctly identified  as low risk, no biopsy necessary: {}".format(round(true_negative_prop,2)))
    print("True positives identified: Proportion of sample correctly classified as high risk, need biopsy: {}".format(round(true_positive_prop, 2)))

    
    false_negative_rate = round(false_negatives / (true_positives + false_negatives),2)
    false_positive_rate = round(false_positives / (false_positives + true_negatives),2)
   
    recall = round(true_positives / (false_negatives + true_positives),2)
    precision = round(true_positives / (false_positives + true_positives),2)
    npp = round(true_negatives/ (true_negatives + false_negatives),2)
    specificity =  round(true_negatives/  (true_negatives + false_positives), 2)
   
     
    print ("Recall / Sensitivity: \n Of all people who actually have metastasis, how many are correctly recommended for biopsy? {}".format(round(recall, 2)))
    print("False negative rate: \n Patients mis-classified as low risk, but need biopsy: {}".format(round(false_negative_rate, 2)))

    print('\n Specificity/True negative rate: \n Of people who are actually low risk, how many are classified as low risk?', format(specificity))
    print("False positive rate: \n Patients classifiedas high risk, but don't need biopsy: {}".format(round(false_positive_rate,2)))

    
    print ("Precision / Positive Predictive Value: \n Of all people recommended for biopsy, how many actually have metastasis? {}".format(round(precision, 2)))
    print('Negative Predictive Value: \n Of people classified as low risk,  how many are actually low risk? {}'.format(round(npp)))

def plot_pr_curve(y_test, y_prob, key_thresh):

    precision, recall, thresh = precision_recall_curve(y_test, y_prob)
    pr_auc_score = auc(recall, precision)
    thresh_lower = np.argmin(np.abs(thresh -key_thresh))
    thresh_default = np.argmin(np.abs(thresh - 0.5))

    plt.figure(figsize = (10, 6))
    plt.step(recall, precision,  where='post',  color = "blue",
        label='PR AUC = %0.2f'% pr_auc_score)
    plt.plot(recall[thresh_lower], precision[thresh_lower], 'o', markersize = 10,
        label = "threshold %0.2f"% key_thresh, mew = 3, color = "red")
    plt.plot(recall[thresh_default], precision[thresh_default], 'o', markersize = 10,
        label = "threshold 0.5", mew = 3, color = "black")
    plt.xlabel('Recall', fontsize = 15)
    plt.ylabel('Precision', fontsize = 15)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve', fontsize = 20)
    plt.legend(loc="lower right")
    plt.tight_layout()


def plot_roc_curve(y_test, y_prob, key_thresh):
    fpr, tpr, thresh = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    thresh_lower = np.argmin(np.abs(thresh - key_thresh))
    thresh_default = np.argmin(np.abs(thresh - 0.5))

    plt.figure(figsize = (10,6))
    plt.step(fpr, tpr,  where='post',  color = "blue",
        label='ROC AUC = %0.2f'% auc_score)
    plt.plot(fpr[thresh_lower], tpr[thresh_lower], 'o', markersize = 10,
            label = "threshold %0.2f"% key_thresh, mew = 3, color = "red")
    plt.plot(fpr[thresh_default], tpr[thresh_default], 'o', markersize = 10,
            label = "threshold 0.5", mew = 3, color = "black")
    plt.plot([0,1], [0,1], color = "black", linestyle='--')
    plt.xlabel('False Positive Rate', fontsize =15)
    plt.ylabel('True Positive Rate', fontsize = 15)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Receiver Operator Curve', fontsize = 20)
    plt.legend(loc="lower right")
    plt.tight_layout()
