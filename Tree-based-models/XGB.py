import pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split
from feature_engineering import *
from sklearn.model_selection import GridSearchCV

categorical_vars = ['subject_sex',
                        
                        'type',
                        'arrest_made',
                        'citation_issued',
                        'outcome',
                        'contraband_found',
                        'contraband_drugs',
                        'warning_issued',
                        'contraband_weapons',
                        'contraband_alcohol',
                        'contraband_other',
                        'frisk_performed',
                        'search_conducted',
                        'search_basis',
                        'reason_for_stop',
                        'vehicle_type',
                        'vehicle_registration_state'
                        
                        
                       
                        ]

for col in categorical_vars:
    data_new[col] = data_new[col].astype("category").cat.codes+1

train, test, y_train, y_test = train_test_split(data_new.drop(["subject_sex"], axis=1), data_new["subject_sex"],
                                                random_state=10, test_size=0.25)



import xgboost as xgb
from sklearn import metrics

def auc(m, train, test): 
    return (metrics.roc_auc_score(y_train,m.predict_proba(train)[:,1]),
                            metrics.roc_auc_score(y_test,m.predict_proba(test)[:,1]))

# Parameter Tuning
model = xgb.XGBClassifier()
param_dist = {"max_depth": [10,30,50],
              "min_child_weight" : [1,3,6],
              "n_estimators": [200],
              "learning_rate": [0.05, 0.1,0.16],}

grid_search = GridSearchCV(model, param_grid=param_dist, cv = 3, 
                                   verbose=10, n_jobs=-1)
grid_search.fit(train, y_train)

print(grid_search.best_estimator_)

# model = xgb.XGBClassifier(max_depth=50, min_child_weight=1,  n_estimators=200,\
#                           n_jobs=-1 , verbose=1,learning_rate=0.16)
# model.fit(train,y_train)

# auc(model, train, test)