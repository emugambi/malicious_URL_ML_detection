#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  4 11:08:10 2025
@author: ernestmugambi
"""
#import os
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

fpath = 'malicious_phish.csv'
data_nrows = 20000
"""
=================================================
string / character manipulation
=================================================
"""
def remove_http(f):
    f = f.dropna()
    new_url = []
    x = ''
    for i in range(len(f)):
        x = f['url'][i]
        cut = x.find('://') + 3
        new_url.append(x[cut:])
    f['new_url'] = new_url
    return f

def remove_sub_domain(f):
    f = f.dropna()
    new_url = []
    x = ''
    for i in range(len(f)):
        x = f['url'][i]
        cut = x.find('/')
        new_url.append(x[:cut])
    f['new_url'] = new_url
    return f
"""
=================================================
feature functions
=================================================
"""
# Calculates the Shannon entropy of a string
def entropy(string):
    # get probability of chars in string
    prob = [ float(string.count(c)) / len(string) for c in dict.fromkeys(list(string)) ]
    # calculate the entropy
    entropy = - sum([ p * math.log(p) / math.log(2.0) for p in prob ])
    return entropy

# count no. of vowels
def vowels(text):
    num_vowels=0
    for char in text:
        if char in "aeiouAEIOU":
           num_vowels = num_vowels+1
    return num_vowels

#count no. of digits
def numbers(text):
    num_numbers=0
    for char in text:
        if char in "1234567890":
           num_numbers = num_numbers+1
    return num_numbers

# count no. of dots
def dots(text):
    num_numbers=0
    for char in text:
        if char in ".":
           num_numbers = num_numbers+1
    return num_numbers

# count no. of forward-slashes
def slashes(text):
    num_numbers=0
    for char in text:
        if char in "/":
           num_numbers = num_numbers+1
    return num_numbers

# count number of symbols found in string
def count_symbols(s):
    t = 0
    symbols = ['!','@','#','$','%','^','&','*','(',')','_','-','+','=','{','}','[',']','\\','//','?','<','>','|','~','`','.',',','/']
    for i in symbols:
        if i in s:
            t += 1
    return t

def split(word): 
    return [char for char in word]  

# what is in a string ?
def categorize_pattern(in_put):
    if in_put.isdigit():                        # numbers 0-9
        return 0
    if in_put.isalpha():                        # letters only
        return 1
    else:
        return 2                                # symbols and combination of symbols/letters/numbers
    
# count pattern types
def count_types(d):
    strings = d.split()
    str_types = []
    for i in strings:
        str_types.append(categorize_pattern(i))
    return (str_types.count(0),str_types.count(1),str_types.count(2),str_types.count(3))
        
# form a transition matrix from chars
def count_transitions(d):
    char_list = split(d)
    res = np.zeros((3, 3))
    for i in range(len(char_list)-1):
        j = i + 1
        res[categorize_pattern(char_list[i]),categorize_pattern(char_list[j])] = res[categorize_pattern(char_list[i]),categorize_pattern(char_list[j])] + 1
    res.resize((1,9))
    return res[0]

# calls all the original feature functions
def get_original_features(data_in):
    features = pd.DataFrame(columns=['shannon','vowels','numbers','dots'])
    s,v,n,d,l,sym,slsh = [],[],[],[],[],[],[]
    for i in range(len(data_in)):
        s.append(np.floor(entropy(data_in['url'][i])))
        #print(i)
        v.append(vowels(data_in['url'][i]))
        n.append(numbers(data_in['url'][i]))
        d.append(dots(data_in['url'][i]))
        l.append(len(data_in['url'][i]))
        sym.append(count_symbols(data_in['url'][i]))
        slsh.append(slashes(data_in['url'][i]))
        #trn.append(count_transitions(data_in['url'][i]))
    features['shannon'] = s
    features['vowels'] = v
    features['numbers'] = n
    features['dots'] = d
    features['length'] = l
    features['symbols'] = sym
    features['f_slash'] = slsh
    return features

# calls the sequence based features
def get_sequences(data_in):
    trn = []
    for i in range(len(data_in)):
        trn.append(count_transitions(data_in['url'][i]))
    df_trn_matrix = pd.DataFrame(data=trn, columns = ['d_d','d_a','d_s','a_d','a_a','a_s','s_d','s_a','s_s'])
    return df_trn_matrix    
"""
=================================================
MAIN : combines all the features together and saves output to a file
=================================================
"""
def generate_features(data):
    #data = get_data_4()        # choose correct data file
    out_a = get_original_features(data)
    out_b = get_sequences(data)
    out = pd.concat([out_a,out_b],axis=1)
    #out.to_csv(feature_output)
    #print("Done....")
    return out

"""
=================================================
data sources
=================================================
"""
def get_train_data(fpath,trn_size):
    all_data = pd.read_csv(fpath)
    # get good training data
    good = all_data[all_data['type'] == 'benign']
    g_sample = np.random.randint(1,len(good),trn_size)
    good_sample = good.iloc[g_sample]
    good_sample = good_sample.drop(['type'],axis = 1)
    good_sample['class'] = 0
    # get bad training data
    bad = all_data[all_data['type'] != 'benign']
    b_sample = np.random.randint(1,len(bad),trn_size)
    bad_sample = bad.iloc[b_sample]
    bad_sample = bad_sample.drop(['type'],axis = 1)
    bad_sample['class'] = 1
    # combined =  good + bad
    train_dat = pd.concat([good_sample,bad_sample],axis=0)
    train_dat = train_dat.reset_index(drop=True)
    return train_dat

# scikit pipeline optimization routine
trn_data = get_train_data(fpath, data_nrows)
train_features = generate_features(trn_data)
X_train, X_test, y_train, y_test = train_test_split(train_features, trn_data['class'], test_size=0.33, random_state=42)
xgb_model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False)
param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.9],
        'colsample_bytree': [0.7, 0.9]
    }
grid_search = GridSearchCV(estimator=xgb_model,
                               param_grid=param_grid,
                               scoring='accuracy',  # Or 'roc_auc', 'neg_log_loss', etc.
                               cv=5,
                               n_jobs=-1,  # Use all available CPU cores
                               verbose=1)
grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_}")
best_xgb_model = grid_search.best_estimator_
test_accuracy = best_xgb_model.score(X_test, y_test)
print(f"Test accuracy of the best model: {test_accuracy}")
"""
=================================================
compute classification performance
=================================================
"""
prediction = best_xgb_model.predict(X_test)
print("confusion matrix:",confusion_matrix(y_test, prediction))
cm = confusion_matrix(y_test, prediction, labels=best_xgb_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_xgb_model.classes_)

## save confusion matrix to file
disp.plot()
plt.savefig("model_results.png", dpi=120)
tn, fp, fn, tp = cm.ravel()
tn, fp, fn, tp = confusion_matrix(y_test, prediction).ravel()
precision, recall = tp/(tp+fp), tp/(tp+fn)
f1 = (2*precision*recall)/(precision+recall)

## Write metrics to file
with open("metrics.txt", "w") as outfile:
    outfile.write(f"\nPrecision = {round(precision, 2)}, recall = {round(precision, 2)},F1 Score = {round(f1, 2)}\n\n")