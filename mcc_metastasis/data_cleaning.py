#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 30 13:50:27 2020
Data cleaning functions for NCDB 
@author: michellevantieghem2
"""
import pandas as pd
import numpy  as np
import math
import warnings

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
        raw data from NCDB on tumor size in mm
        some data is binned, some data indicates range.

    Returns
    -------
    recode_var : variable (float)
        all continuous exact values (mm)converted into cm
        values provided as bins (start with 99) are converted to corresponding value in cm
    """
    
    if (var == 0):
        recode_var = np.nan
    # if exactt value provided, convert from mm to cm
    elif (var < 990):
        recode_var = var / 10
    # otherwise, convertt bins to corresponding cm values 
    elif (var == 991) :
        recode_var = 1
    elif (var == 992) :
        recode_var = 2
    elif (var == 993) :
        recode_var = 3
    elif (var == 994) :
        recode_var = 4
    elif (var == 995) :
        recode_var = 5
    elif(var == 996) :
        recode_var = 6 # but this really means 5 +
    # everything else is nan
    else:
        recode_var = np.nan
    return recode_var


#def recode_outcome (df):
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
    # if they have both variables...  
   # if (not math.isnan(df.regional_nodes_positive_bin)) & (not math.isnan(df.lymph_node_mets_bin)):
        # provide value if they both agree
   #     if (df.regional_nodes_positive_bin == df.lymph_node_mets_bin): 
    #        return df.regional_nodes_positive_bin
       
        # if they don't agree, exclude as NAN
     #   else: 
      #      return np.nan 
        
    # if they only have one variable, use that
    #elif (not math.isnan(df.regional_nodes_positive_bin)):
     #   return df.regional_nodes_positive_bin
    #elif (not math.isnan(df.lymph_node_mets_bin)):
     #   return df.lymph_node_mets_bin

# %% Making tables 

def get_missingness_table(df):
    
    """ Calculate missingness in dataframe

    Parameters
    ----------
    df : pandas dataframe

    Returns
    -------
    missing_table : dataframe
        provides total missing cases per variable in dataframe

    """
    missing_table = (
        df.isnull()
        .sum()
        .to_frame("Missing")
        .reset_index()
    )
    print("Missingness of features")
    return missing_table

def get_N_table(df, var):
    
    """ Make table that displays N per group 
    

    Parameters
    ----------
    df : pandas dataframe
    var : grouping variable

    Returns
    -------
    table : N per group in a nice table
        

    """
    table =  (
        df
        .groupby(var, dropna = False)
        .size()
        .to_frame('N')
        .reset_index()
    )
    return table

def get_proportions(data, var):
    
    """
    

    Parameters
    ----------
    data : TYPE
        DESCRIPTION.
    var : TYPE
        DESCRIPTION.

    Returns
    -------
    counts : TYPE
        DESCRIPTION.

    """
    counts = pd.DataFrame(np.unique(data, return_counts = True), index = [var, 'Counts']).T
    counts['Percent'] = (counts['Counts']/counts['Counts'].sum()).round(2)*100
    counts  = counts.set_index(var)
    return counts 

 