#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 10 14:39:51 2020
Plotting functions 
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

# custom plot code 
def plot_predprobs(y_probs, key_thresh, filename):
    plt.figure(figsize = (10, 8))
    plt.hist(y_probs, alpha = 0.5)
    plt.axvline(linewidth = 4, color = "red", linestyle = "dotted",
                x = key_thresh, label = "Thresh %0.2f" % key_thresh)
    plt.axvline(linewidth = 4, color = "black", linestyle = "dotted",
                x = 0.5, label = "Thresh 0.5")
    plt.yticks(fontsize = 12)
    plt.xticks(fontsize = 12)
    plt.xlim(0, 1)
    plt.xlabel("Predicted Probability of Metastasis", size = 15)
    plt.ylabel ("Frequency",  size = 15)
    plt.legend(loc="center right", ncol=1)
    plt.savefig(filename, bbox_inches = 'tight')
    

def plot_pr_curve(y_test, y_prob, key_thresh, filename):

    precision, recall, thresh = precision_recall_curve(y_test, y_prob)
    pr_auc_score = auc(recall, precision)
    
    precision_default = precision_score(y_test, y_prob > 0.5)
    recall_default = recall_score(y_test, y_prob > 0.5)
    precision_key = precision_score(y_test, y_prob > key_thresh)
    recall_key = recall_score(y_test, y_prob > key_thresh)

    plt.figure(figsize = (10, 6))
    plt.step(recall, precision,  where='post',  color = "blue",
        label='PR AUC = %0.2f'% pr_auc_score)
    plt.plot(recall_key, precision_key, 'o', markersize = 10,
        label = "Thresh: %0.2f \nPrecision: %0.2f \nRecall: %0.2f" % (key_thresh, precision_key, recall_key), 
             mew = 3, color = "red")
    plt.plot(recall_default, precision_default, 'o', markersize = 10,
       label = "Thresh: %0.2f \nPrecision: %0.2f \nRecall: %0.2f" % (0.5, precision_default, recall_default), 
              mew = 3, color = "black")
    plt.xlabel('Recall', fontsize = 20)
    plt.ylabel('Precision', fontsize = 20)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve', fontsize = 20)
    plt.legend(loc="lower right", fontsize = 16)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight')



def plot_roc_curve(y_test, y_prob, key_thresh, filename):
    fpr, tpr, thresh = roc_curve(y_test, y_prob)
    auc_score = auc(fpr, tpr)
    thresh_lower = np.argmin(np.abs(thresh - key_thresh))
    thresh_default = np.argmin(np.abs(thresh - 0.5))

    plt.figure(figsize = (10,6))
    plt.step(fpr, tpr,  where='post',  color = "blue",
        label='ROC AUC = %0.2f'% auc_score)
    plt.plot(fpr[thresh_lower], tpr[thresh_lower], 'o', markersize = 10,
            label = "Thresh: %0.2f \n FPR: %0.2f \nTPR: %0.2f"% (key_thresh, fpr[thresh_lower], tpr[thresh_lower]),
                                                                 mew = 3, color = "red")
    plt.plot(fpr[thresh_default], tpr[thresh_default], 'o', markersize = 10,
            label = "Thresh: %0.2f \n FPR: %0.2f \nTPR: %0.2f"% (0.5, fpr[thresh_default], tpr[thresh_default]),
             mew = 3, color = "black")
    plt.plot([0,1], [0,1], color = "black", linestyle='--')
    plt.xlabel('False Positive Rate', fontsize = 20)
    plt.ylabel('True Positive Rate', fontsize = 20)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Receiver Operator Curve', fontsize = 20)
    plt.legend(loc="lower right", fontsize = 16)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches = 'tight')



# https://medium.com/analytics-vidhya/probability-calibration-essentials-with-code-6c446db74265
def plot_calibration_curve(est, name, method, fig_index, X_train, X_test, y_train, y_test, filename):
    """Plot calibration curve for est w/o and with calibration. """
    # Calibrated with isotonic calibration
    calibrator = CalibratedClassifierCV(est, cv=2, method= method)

    fig = plt.figure(fig_index, figsize=(10, 10))
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax2 = plt.subplot2grid((3, 1), (2, 0))

    ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
    for clf, name in [(est, name),
                      (calibrator, name + ' +' + method)]:
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
    plt.savefig(filename, bbox_inches = 'tight')
  
  
def plot_coefs (coefs_df, filename): 
    n_features =  coefs_df.shape[0]
    plt.figure(figsize =  (8,6))
    g = sns.barplot(data = coefs_df, y = 'features', x = "coefs",
            palette = sns.color_palette("vlag", n_features))
    g.set_ylabel("")
    g.tick_params(labelsize = 15)
    g.set_xlabel("Coefficients", size = 15)
    g.set_title("Feature importance", size = 20);
    plt.savefig(filename, bbox_inches = 'tight')
