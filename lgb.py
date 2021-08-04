# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 18:38:29 2020

@author: doggy
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import warnings
import time

#from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from bayes_opt import BayesianOptimization
from preprocess import *


def make_data():
    #分層採樣，確保訓練集與測試集中各類樣本的比例與原始數據集中相同
    cat_cols, users, X, y = import_and_create_train_valid_data('application_train.csv.zip')
    X_test, test_users = import_and_create_test_data('application_test.csv.zip')
    
    bayesian_train_index, bayesian_val_index = list(StratifiedKFold(n_splits=2).split(X, y))[0] #random_state=1
    #print('\ntrain_size: %s | val_size: %s' % (bayesian_train_index.shape, bayesian_val_index.shape))
 
    return cat_cols, users, X, y, \
        X_test, test_users, bayesian_train_index, bayesian_val_index
    
    
    
def LGB_bayesian(max_depth, num_leaves, learning_rate,
                 reg_alpha, reg_lambda, bagging_fraction, bagging_freq, feature_fraction): 
    
    cat_cols, _, X, y, _, _, bayesian_train_index, bayesian_val_index = make_data()
        
    parameters = {'boosting_type': 'gbdt',          #設置提升類型
                 'objective': 'binary',             #目標函數
                 'n_estimators': 200,
                 'learning_rate':learning_rate,
                 'max_depth': int(max_depth),       #if -1 means no limit
                 'n_jobs': -1,
                 'num_leaves': int(num_leaves),     #葉子節點數 (we should let it be smaller than 2^(max_depth))
                 'objective': 'binary', 
                 'random_state': 1, 
                 'reg_alpha': reg_alpha, 
                 'reg_lambda': reg_lambda,  
                 'is_unbalance': True,
                 'metric': 'auc',                      #('auc','binary_logloss')
                 'verbose': 2,
                 'feature_fraction': feature_fraction, #特徵採樣
                 'bagging_fraction': bagging_fraction, #數據採樣
                 'boost_from_average': False,
                 'bagging_freq': int(bagging_freq)     #每K輪迭代執行一次bagging
                 #'scale_pos_weight':11
                 }
    
    lgb_train = lgb.Dataset(X.iloc[bayesian_train_index], y.iloc[bayesian_train_index])
    lgb_valid = lgb.Dataset(X.iloc[bayesian_val_index], y.iloc[bayesian_val_index])
    
    #clf = gb.cv(param, train_data, num_round, nfold=5)
    clf = lgb.train(parameters, lgb_train, valid_sets = lgb_valid, early_stopping_rounds = 30,
                    categorical_feature = cat_cols)
    
    predictions = clf.predict(X.iloc[bayesian_val_index], num_iteration =clf.best_iteration) 
    score = roc_auc_score(y.iloc[bayesian_val_index], predictions) 
    
    return score 


def LGB_bayesian_train():
    bounds_LGB = {
    'num_leaves': (32, 1024), 
    'learning_rate': (0.001, 0.1),
    'feature_fraction': (0.7, 1.0),
    'bagging_fraction': (0.7, 1.0),
    'bagging_freq': (5, 20),
    'reg_alpha': (0, 1),
    'reg_lambda': (0, 50),
    'max_depth':(5, 10)}
    
    print('\n\nfinding best parameters by Bayesian Optimization.....\n\n')
    LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=1)
    
    #創建BayesianOptimization对象(LGB_BO)後，在調用maxime前不會運作。
    #init_points：How many steps of random exploration you want to perform.(help by diversifying the exploration space.)
    #n_iter：How many steps of bayesian optimization you want to perform.(The more steps the more likely to find a good maximum)
    init_points = 10
    n_iter = 50

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        LGB_BO.maximize(init_points =init_points, n_iter =n_iter)
        
    print('\n\nmax valid score: %d \n\n' % (LGB_BO.max['target']))
    
    #保存最佳參數
    best_parameters = { 'boosting_type': 'gbdt', # ['dart', 'gbdt', 'goss']         
                        'objective': 'binary',   
                        'n_estimators': 2000,
                        'class_weight': None, 
                        'n_jobs': -1,
                        'objective': 'binary', 
                        'random_state': 1,  
                        'is_unbalance': True,
                        'metric': 'auc', 
                        'verbose': 2,
                         #'scale_pos_weight': 11,
                        'learning_rate': LGB_BO.max['params']['learning_rate'],
                        'num_leaves': int(LGB_BO.max['params']['num_leaves']),
                        'feature_fraction': LGB_BO.max['params']['feature_fraction'],
                        'bagging_fraction': LGB_BO.max['params']['bagging_fraction'],  
                        'bagging_freq': int(LGB_BO.max['params']['bagging_freq']), 
                        'reg_alpha':  LGB_BO.max['params']['reg_alpha'], 
                        'reg_lambda': int(LGB_BO.max['params']['reg_lambda']), 
                        'max_depth': int(LGB_BO.max['params']['max_depth']),
                        'boost_from_average': False
                         #'min_split_gain':  LGB_BO.max['params']['min_split_gain'],
                         #'min_data_in_leaf': int(LGB_BO.max['params']['min_data_in_leaf']),
                         #'min_sum_hessian_in_leaf': LGB_BO.max['params']['min_sum_hessian_in_leaf'],
                         #'drop_seed':int(2**5)
                         }
                         
    return best_parameters


if __name__ == '__main__':
    cat_cols, _, X, y, X_test, test_users, _, _ = make_data()  
    best_parameters = LGB_bayesian_train()
    
    nfold = 5
    skf = StratifiedKFold(n_splits = nfold, shuffle=True, random_state=1) 
    
    val_predict = np.zeros(len(X))
    predictions = np.zeros((len(X_test), nfold))
    print("\n\nStart training Final model......\n\n")
    
    tStart = time.time()#計時開始
    i = 0
    for train_index, valid_index in skf.split(X, y):
        print("\nfold {}".format(i+1))
        lgb_train = lgb.Dataset(X.iloc[train_index],
                                y.iloc[train_index])
        
        lgb_valid = lgb.Dataset(X.iloc[valid_index],
                                y.iloc[valid_index])
        
        evals_result = {} #紀錄訓練結果
        clf = lgb.train(best_parameters, lgb_train, valid_sets = lgb_valid,
                        early_stopping_rounds = 50, evals_result=evals_result,
                        categorical_feature = cat_cols)
        
        val_predict[valid_index] = clf.predict(X.iloc[valid_index], num_iteration = clf.best_iteration)
        predictions[:,i] += clf.predict(X_test[X.columns], num_iteration = clf.best_iteration)
        i += 1
    
    tEnd = time.time()#計時結束
    print("\n\n AUC in train data: %.4f" % (roc_auc_score(y, val_predict)))
    print ("It cost %f sec" % (tEnd - tStart))#會自動做進位

    predictions_transform = sum(predictions.T) / nfold
    submission = pd.DataFrame({'SK_ID_CURR' : test_users['SK_ID_CURR'], 'TARGET' :predictions_transform})
    submission.to_csv('submission_lgb.csv', index=False) 

