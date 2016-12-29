#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 12:56:47 2016

@author: meghanasatish
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 20 18:25:41 2016

@author: meghanasatish
"""

import pandas as pd
import statsmodels.api as sm
import pylab as pl
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import sklearn.metrics
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from matplotlib.pyplot import *
import sqlite3
from sklearn import svm

#con = sqlite3.connect('../input/database.sqlite')
#loan = pd.read_csv('../input/loan.csv')

df = pd.read_csv("loan.csv", low_memory = False)
print(df)

#Reading into the data

relevant_data=['member_id'	,'loan_amnt','purpose','funded_amnt','funded_amnt_inv'	,'term','int_rate','installment','home_ownership','grade','sub_grade'	,'emp_length'	,'annual_inc'	,'loan_status','verification_status','addr_state','dti','earliest_cr_line','open_acc','revol_bal','revol_util','total_acc','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','last_pymnt_amnt','last_credit_pull_d']

loan_necessary=df[relevant_data]
loan_necessary.to_csv("loans_necessary.csv")

#Cleaning up the data

loan_necessary['term'] = loan_necessary['term'].str.rstrip('months')
print(term)

loan_necessary['emp_length'] = loan_necessary['emp_length'].str.extract("(\d+)").map(float)
print(emp_length)

loan_necessary[["loan_amnt","annual_inc"]].dropna().describe()

loan_necessary["int_rate"]=loan_necessary["int_rate"].astype(float)

loan_necessary.isnull().sum()

loan_necessary = loan_necessary.dropna(subset=['member_id','loan_amnt','purpose','funded_amnt','funded_amnt_inv'	,'term','int_rate','installment','grade','sub_grade'	,'emp_length'	,'annual_inc'	,'loan_status','verification_status','addr_state','dti','earliest_cr_line','open_acc','revol_bal','revol_util','total_acc','total_pymnt','total_pymnt_inv','total_rec_prncp','total_rec_int','last_pymnt_amnt','last_credit_pull_d'])

#Plotting 

newdf = loan_necessary[['loan_amnt','purpose']].groupby('purpose').mean()
print (newdf)
newdf.plot(kind='bar')

newdf1 = loan_necessary[['loan_amnt','loan_status']].groupby('loan_status').count()
print (newdf1)
newdf1.plot(kind='bar')

newdf3 = loan_necessary[['loan_amnt','emp_length']].groupby('emp_length').count()
newdf3.plot(kind='bar')

newdf1 = loan_necessary[['loan_amnt','addr_state']].groupby('addr_state').mean()
print (newdf1)

data=loan_necessary[['loan_amnt','addr_state','purpose']].groupby('addr_state').mean().count()
print(data)
newdf4 = loan_necessary[['loan_amnt','loan_status']].groupby('loan_status').count()
newdf4.plot(kind='bar')


loan_necessary['verification_status']=loan_necessary['verification_status'].replace('Source Verified', 'Verified')
loan_necessary['verification_status'] = pd.Series(loan_necessary['verification_status'], dtype="category")

loan_necessary['verification_status']=loan_necessary['verification_status'].cat.rename_categories([0,1])

print(loan_necessary['verification_status'])


loan_necessary=loan_necessary[loan_necessary.loan_status !='Current']
loan_necessary=loan_necessary[loan_necessary.loan_status !='In Grace Period']
loan_necessary=loan_necessary[loan_necessary.loan_status !='Issued']
loan_necessary=loan_necessary[loan_necessary.loan_status !='Late (16-30 days)']
loan_necessary=loan_necessary[loan_necessary.loan_status !='Late (31-120 days)']              
loan_necessary=loan_necessary[loan_necessary.loan_status !='Default']  

loan_necessary['loan_status']=loan_necessary['loan_status'].replace('Does not meet the credit policy. Status:Fully Paid','Fully Paid')#loan_necessary['loan_status']=loan_necessary['loan_status'].replace('Does not meet the credit policy. Status:Charged Off','Charged Off')

                                
loan_necessary['loan_status'] = pd.Series(loan_necessary['loan_status'], dtype="category")

loan_necessary['loan_status']=loan_necessary['loan_status'].cat.rename_categories([0,1])

print(loan_necessary['loan_status'])

Predictors=loan_necessary[['loan_amnt','term','dti','emp_length','verification_status']]
Target=loan_necessary['loan_status']

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Predictors, Target, test_size=0.2, random_state=42)

#Define a simple train-predict utility function
def train_predict(classification, X_train, X_test, y_train, y_test):

    classification.fit(X_train, y_train)
   
   y_pred = clf.predict(X_test)
    return y_pred
    
from sklearn.tree import DecisionTreeClassifier
classification = DecisionTreeClassifier(random_state=42)
y_pred = train_predict(classification, X_train, X_test, y_train, y_test)

print(classification_report(y_test, y_pred))

print(sklearn.metrics.r2_score(y_pred,y_test))
print(sklearn.metrics.mean_squared_error(tar_test,predictions))

#SVM

clf = svm.SVC(kernel='linear', C = 1.0)
clf.fit(X,y)
y_pred = train_predict(clf, X_train, X_test, y_train, y_test)
print(classification_report(y_test, y_pred))
y_pred = train_predict(clf, X_train, X_test, y_train, y_test)
print(classification_report(y_test, y_pred))



