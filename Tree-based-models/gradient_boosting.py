'''
Created By:  Anirudh Sharma 
Implementations: Light GBM for multi-class classification
Description: This model would like to predict the race of the person stopped in massachusetts based on 
             stanford's open pilicing data set (link - )
'''

# Importing necessary libaries for implementation 
import numpy as np 
import pandas as pd 
import lightgbm as lgb 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.metrics import mean_squared_error, roc_auc_score,precision_recall_curve,precision_score,roc_curve
from sklearn.model_selection import GridSearchCV
from feature_engineering import *



categorical_vars = ['subject_sex','type','arrest_made','citation_issued','outcome','contraband_found','contraband_drugs','warning_issued','contraband_weapons','contraband_alcohol','contraband_other','frisk_performed','search_conducted','search_basis','reason_for_stop','vehicle_type','vehicle_registration_state']


for categorical in list(data_new.columns):
    if categorical in categorical_vars:
        data_new[categorical] = data_new[categorical].astype('category').cat.codes+1



print(data_new.dtypes)

train, test, y_train, y_test = train_test_split(data_new.drop(["subject_race"], axis=1), data_new["subject_race"],
                                                random_state=10, test_size=0.25)

def auc_1(model,train,test):
    return (metrics.roc_auc_score(y_train,model.predict(train)),
            metrics.roc_auc_score(y_test,model.predict(test)))

# light_g = lgb.LGBMClassifier(silent= False)
# parameter_list = {"max_depth": [25,50, 75],
#               "learning_rate" : [0.01,0.05,0.1],
#               "num_leaves": [300,900,1200],
#               "n_estimators": [200]
#              }
# grid_search = GridSearchCV(light_g, n_jobs=-1, param_grid=parameter_list, cv = 3, scoring="roc_auc", verbose=5)
# grid_search.fit(train,y_train)
# print(grid_search.best_estimator_)


training_set = lgb.Dataset(train,label = y_train)

# training with categorical features

cat_features = ["subject_sex",
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
                'vehicle_registration_state']
parameters = {"learning_rate": 0.05,"max_depth": 50, "n_estimators": 200, "num_leaves" : 300, "silent" : False}
model = lgb.train(parameters, training_set, categorical_feature = categorical_vars)
auc_score_fortrain, auc_score_for_test = auc_1(model,train,test)
print(auc_score_fortrain)
print(auc_score_for_test)
