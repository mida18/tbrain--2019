# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 16:39:28 2020

@author: doggy
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings

from preprocess import *
# from sklearn.metrics import roc_auc_score
from featexp import get_trend_stats

def make_data():
    warnings.filterwarnings("ignore")
    print('train data：')
    X_train, X_valid, y_train, y_valid, train_users, valid_users = \
        import_and_create_train_valid_data(data ='application_train.csv.zip')
    print('\ntest data：')
    X_test, test_users = import_and_create_test_data(data ='application_test.csv.zip')
    
    #drop=['CODE_GENDER_XNA', 'NAME_INCOME_TYPE_Maternity leave', 'NAME_FAMILY_STATUS_Unknown']
    #X_train = X_train.drop(drop, axis=1)
    #X_valid = X_valid.drop(drop, axis=1)
    
    data_train = X_train.reset_index(drop=True)
    data_train['target'] = y_train.reset_index(drop=True)
    
    data_valid = X_valid.reset_index(drop=True)
    data_valid['target'] = y_valid.reset_index(drop=True)
    
    return X_train, X_valid, y_train, y_valid, train_users, valid_users, \
           X_test, test_users, \
           data_train, data_valid
    
def model_all_features():
    X_train, X_valid, y_train, y_valid, train_users, valid_users, \
           X_test, test_users, _, _ = make_data()
  
    dvalid = xgb.DMatrix(X_valid, label=y_valid, missing=np.nan)
    dtrain = xgb.DMatrix(X_train, label=y_train, missing=np.nan)
    
    best_parameters = {'max_depth':10, 'learning_rate':0.01, 'silent':0, 'objective':'binary:logistic', 
                       'min_child_weight':600, 'eval_metric' : 'auc', 'scale_pos_weight':11, 
                       'colsample_bytree':0.8, 'missing' : np.nan, 'random_state': 0 } 
    #scale_pos_weight = len(df[df['TARGET']== 0]) / len(df[df['TARGET']== 1]) = 11.387150050352467
    
    warnings.filterwarnings("ignore")
    xgb_model = xgb.train(best_parameters, dtrain, 1000, evals=[(dtrain, 'train'), (dvalid, 'test')], early_stopping_rounds=25)
    
    dtest = xgb.DMatrix(X_test[X_valid.columns], missing=np.nan)
    y_test_pred = xgb_model.predict(dtest)
    submission_all_feats = pd.DataFrame({'SK_ID_CURR' : test_users['SK_ID_CURR'], 'TARGET' : y_test_pred})
    
    return submission_all_feats

def model_filter(corr_threshold = 0.90, importance = 10):
    X_train, X_valid, y_train, y_valid, train_users, valid_users, \
           X_test, test_users, data_train, data_valid = make_data()
    
    # trend correlations & feature importance
    stats = get_trend_stats(data=data_train, target_col='target', data_test=data_valid)
    importance_df = get_imp_df(xgb_model) # get xgboost importances in dataframe
    stats = pd.merge(stats, importance_df, how='left', on='Feature')
    stats['importance'] = stats['importance'].fillna(0)
    
    # Dropping features with trend corr < 0.97 / 0.95 / 0.93 / 0.90
    noisy = list(stats[stats['Trend_correlation'] < corr_threshold]['Feature'])
    
    # Dropping features with trend corr & feature importance
    #noisy = list(stats[(stats['Trend_correlation'] < corr_threshold) & \
    #                   (stats['importance'] < importance)]['Feature'])

    dvalid = xgb.DMatrix(X_valid.drop(noisy, axis=1), label=y_valid, missing=np.nan)
    dtrain = xgb.DMatrix(X_train.drop(noisy, axis=1), label=y_train, missing=np.nan)
    
    params =  {'max_depth':10, 'learning_rate':0.01, 'silent':0, 'objective':'binary:logistic', 
                       'min_child_weight':600, 'eval_metric' : 'auc', 'scale_pos_weight':11, 
                       'colsample_bytree':0.8, 'missing' : np.nan, 'random_state': 0 } 
    
    xgb_model = xgb.train(params, dtrain, 1000, evals=[(dtrain, 'train'), (dvalid, 'test')], early_stopping_rounds=25)

    dtest = xgb.DMatrix(X_test[X_valid.columns].drop(noisy, axis=1), missing=np.nan)
    y_test_pred = xgb_model.predict(dtest)
    submission = pd.DataFrame({'SK_ID_CURR' : test_users['SK_ID_CURR'], 'TARGET' : y_test_pred})
    
    return submission
 

if __name__ == '__main__':

    submission_all_feats = model_all_features()
    submission_all_feats.to_csv('submission_all_feats.csv', index=False) 
    
    submission = model_filter(corr_threshold = 0.90, importance = 10)
    submission.to_csv('submission.csv', index=False)
