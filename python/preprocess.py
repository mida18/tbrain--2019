# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
#from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor


id_col = "SK_ID_CURR"
target_col = "TARGET"

# KNN Imputer
def find_no_nan_columns(df):
    remain_nan_cols = df.isnull().any() [df.isnull().any()== True].index.tolist()
    all_cols =  df.columns.tolist()
    all_cols.remove('SK_ID_CURR')
    try:
        all_cols.remove('TARGET')
    except:
        pass
    
    no_nan_cols = list(set(all_cols) - set(remain_nan_cols))
    return no_nan_cols

def knn_train_test(target_col, df, no_nan_cols):
    train_y = df[target_col][df[target_col].isnull() == False] #目標欄位下的非NAN值
    train_x = df[df[target_col].isnull() == False].loc[:,no_nan_cols] #行列都沒有NAN值者
    test_x = df[df[target_col].isnull() == True].loc[:,no_nan_cols] #目標欄位下含NAN值者的其餘非NAN列值
    return train_y, train_x, test_x

def knn_missing_filled(train_x, train_y, test_x, k = 3, dispersed = True):
    if dispersed:
        clf = KNeighborsClassifier(n_neighbors = k, weights = "distance")
    else:
        clf = KNeighborsRegressor(n_neighbors = k, weights = "distance")
    
    clf.fit(train_x, train_y)
    return test_x.index, clf.predict(test_x)

def filled_pred_to_data(target_columns, df , n=3, data_type=True):
    for i in target_columns:
        no_nan_cols = find_no_nan_columns(df)
        train_y, train_x, test_x =  knn_train_test(i, df, no_nan_cols)
        index, pred = knn_missing_filled( train_x, train_y, test_x, k = n, dispersed = data_type)
        df.loc[index, i] = pred
    return df

    
def get_nonull_data(application_train_raw, dummy_drop=[]):
   
    # object：dummy (make NaN as a category)
    # Notice：先注意是否有「數值」型態誤被設定為「類別」型態的欄位
    if len(dummy_drop) != 0:
        application_train_raw.drop(columns=dummy_drop, axis=1, inplace=True) 
        
    cat_cols = application_train_raw.select_dtypes(include="O").columns.tolist()
       
    # if model == 'lgb':
    # 把 category data中 np.nan 替換成 string'NaN'，使其作為一個類別一同被 encode為數字
    for i in cat_cols:
        if application_train_raw[i].isna().values.sum() > 0:
            application_train_raw[i] = application_train_raw[i].fillna('NaN')
        labelencoder = LabelEncoder()
        application_train_raw[i] = labelencoder.fit_transform(application_train_raw[i])
            
    # if model == 'xgb':
    '''
    # 把 category 轉換成 dummy variables，nan也成為一個類別(欄位)
    application_train_raw = pd.get_dummies(
        application_train_raw, columns=cat_cols, drop_first=True, dummy_na=True)
    '''
    
    #print('\n# of cat_cols： %d ' % len(cat_cols))

    
    # float：null data _ less (impute with mean / KNN)
    threshold = 3075
    nulls = pd.isnull(application_train_raw).sum()
    less_nulls = nulls[(nulls < threshold) & (nulls != 0)].index
    less_nulls_float = application_train_raw[less_nulls].select_dtypes(exclude="O").columns

    #print('# of less_nullls_float： %d ' % len(less_nulls_float))
    
    application_train_raw[less_nulls_float] = application_train_raw[
        less_nulls_float].fillna(application_train_raw[less_nulls_float].mean())
    
    '''
    # KNN
    application_train_raw = filled_pred_to_data(less_nulls_float, application_train_raw, n=3, data_type=False)
    '''
    
    # float：null data _ more (impute with large negative number)
    more_nulls = nulls[(nulls >= threshold)].index
    more_nulls_float = application_train_raw[more_nulls].select_dtypes(exclude="O").columns

    #print('# of more_nullls_float： %d ' % len(more_nulls_float))
    
    application_train_raw[more_nulls_float] = application_train_raw[
        more_nulls_float].fillna(application_train_raw[more_nulls_float].min() - 100)

    return application_train_raw, cat_cols


def import_and_create_train_valid_data(data):
    application_raw = pd.read_csv(data, compression='zip', header=0, sep=',', quotechar='"')
    #application_raw = pd.read_csv(data, encoding='big5', low_memory=False)
          
    # if model == 'lgb':     
    application, cat_cols = get_nonull_data(application_raw, dummy_drop=[])
    users = application[[id_col]] 
    X = application.drop([target_col, id_col], axis=1) # id column & y column
    y = application[target_col]
    
    return (cat_cols, users, X, y)

    # if model == 'xgb':     
    '''
    X = application.drop([target_col], axis=1)  # Contains ID.
    y = application[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y)

    train_users = X_train[[id_col]]  # [[]] -> dataframe格式
    train_users[target_col] = y_train
    test_users = X_test[[id_col]]
    test_users[target_col] = y_test
    train_users.reset_index(drop=True, inplace=True)
    test_users.reset_index(drop=True, inplace=True)
    
    X_train = X_train.drop([id_col], axis=1)
    X_test = X_valid.drop([id_col], axis=1)

    return (X_train, X_test, y_train, y_test, train_users, test_users)
    '''

def import_and_create_test_data(data):
    application_raw = pd.read_csv(data, compression='zip', header=0, sep=',', quotechar='"')
    #application_raw = pd.read_csv(data, encoding='big5', low_memory=False)
    application, _ = get_nonull_data(application_raw, dummy_drop=[])

    X = application  # Contains ID.

    users = X[[id_col]]
    users.reset_index(drop=True, inplace=True)

    return (X, users)

def get_imp_df(xgb_model):
    imp = pd.DataFrame(np.asarray(list(xgb_model.get_fscore().keys()))) # Get feature importance of each feature.
    imp.columns = ["Feature"]
    imp["importance"] = np.asarray(list(xgb_model.get_fscore().values()))
    imp = imp.sort_values(by=["importance"], ascending=False)
    imp = imp.reset_index(drop=True)
    return imp
        
    