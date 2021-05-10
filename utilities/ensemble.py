import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr
from math import sqrt
import pickle
import os
import numpy as np
import scipy.stats as sp
import math
from sklearn.model_selection import train_test_split


# most of the codes below is modification of the D3R evaluation codes


B1 = pd.read_csv("test_output", names=['predicted', 'label']) # Load csv
B2 = pd.read_csv("test_output2", names=['predicted', 'label']) # Load csv
B3 = pd.read_csv("test_output3", names=['predicted', 'label']) # Load csv
B4 = pd.read_csv("test_output4", names=['predicted', 'label']) # Load csv
B5 = pd.read_csv("test_output5", names=['predicted', 'label']) # Load csv

x1 = np.array(B1[:]["predicted"])
yall = np.array(B1[:]["label"])
x2 = np.array(B2[:]["predicted"])
x3 = np.array(B3[:]["predicted"])
x4 = np.array(B4[:]["predicted"])
x5 = np.array(B5[:]["predicted"])
xall = []
for i, ele in enumerate(x1):
    tmp = ele + x2[i] + x3[i] + x4[i] + x5[i]
    tmp = tmp/5
    xall.append(tmp)

print("RMSE for 1:", sqrt(mean_squared_error(yall, x1)))
print("RMSE for 2:", sqrt(mean_squared_error(yall, x2)))
print("RMSE for 3:", sqrt(mean_squared_error(yall, x3)))
print("RMSE for 4:", sqrt(mean_squared_error(yall, x4)))
print("RMSE for 5:", sqrt(mean_squared_error(yall, x5)))
print("RMSE for all:", sqrt(mean_squared_error(yall,xall)))
#===========================================


def rms(predictions, targets):
    #calculate the centered rms
    diff = predictions - targets
    mean_diff = np.mean(diff)
    translated_diff = diff - mean_diff
    return np.sqrt((translated_diff**2).mean())
    
    
def bootstrap_exptnoise( calc1, expt1, exptunc1 = 0):
    #using bootstrap to generate new data 
    calc = np.array(calc1)
    expt = np.array(expt1)
    exptunc = np.array(exptunc1)
    npoints = len(calc)
    idx = np.random.randint( 0, npoints, npoints)
    newcalc = calc[idx]
    newexpt = expt[idx]                                             
    newexptunc = exptunc[idx]
    if exptunc == []:
        noise = np.zeros( npoints)
    else:
        noise = np.random.normal(0., exptunc, npoints)             
    newexpt += noise                                                
    return newcalc, newexpt 
    
    
def calculate_kendalls (template_value, submitted_value, exp_unc=[], boot_bins = 10000):
    #template_value, submitted_value should be list type
    #calculating kendalls tau etc using bootstrapping resampling method
    if len(template_value)> 2:
        taus = np.zeros(boot_bins)
        spearmans = np.zeros(boot_bins)
        rms_errors = np.zeros(boot_bins)
        Pearsons = np.zeros(boot_bins)
        for i in range (boot_bins):
            new_template_value, new_submitted_value = template_value, submitted_value#bootstrap_exptnoise(template_value, submitted_value, exp_unc)
            rms_errors[i] = rms(new_template_value, new_submitted_value)
            taus[i] = sp.kendalltau(new_template_value, new_submitted_value)[0]
            if math.isnan(taus[i]):
                taus[i] = 0
            spearmans[i] = sp.spearmanr(new_template_value, new_submitted_value)[0]
            Pearsons[i] = sp.pearsonr(new_template_value, new_submitted_value)[0]
        rms_error = rms(np.asarray(template_value), np.asarray(submitted_value))
        tau = sp.kendalltau(template_value, submitted_value)[0]
        spearman = sp.spearmanr(template_value, submitted_value)[0]
        Pearson = sp.pearsonr(template_value, submitted_value)[0]
        return (rms_error, rms_errors.std(), tau, taus.std(), spearman, spearmans.std(), Pearson, Pearsons.std())
    else:
        return False

rms_error, _, tau, _, spearman, _, Pearson, _ = calculate_kendalls(yall, x1)
print("For 1, rms_error, tau, spearman, Pearson: ",rms_error, tau, spearman, Pearson)
rms_error, _, tau, _, spearman, _, Pearson, _ = calculate_kendalls(yall, x2)
print("For 2, rms_error, tau, spearman, Pearson: ",rms_error, tau, spearman, Pearson)
rms_error, _, tau, _, spearman, _, Pearson, _ = calculate_kendalls(yall, x3)
print("For 3, rms_error, tau, spearman, Pearson: ",rms_error, tau, spearman, Pearson)
rms_error, _, tau, _, spearman, _, Pearson, _ = calculate_kendalls(yall, x4)
print("For 4, rms_error, tau, spearman, Pearson: ",rms_error, tau, spearman, Pearson)
rms_error, _, tau, _, spearman, _, Pearson, _ = calculate_kendalls(yall, x5)
print("For 5, rms_error, tau, spearman, Pearson: ",rms_error, tau, spearman, Pearson)
rms_error, _, tau, _, spearman, _, Pearson, _ = calculate_kendalls(yall, xall)
print("For all, rms_error, tau, spearman, Pearson: ",rms_error, tau, spearman, Pearson)
