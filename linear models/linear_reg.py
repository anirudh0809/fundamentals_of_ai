import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression

data = pd.read_csv('/Users/anirudhsharma/FAI/ma_statewide_2020_04_01.csv', low_memory = False)
print(data.head(10))

# print(data['search_conducted'].unique())
# print(data['outcome'].unique())

# print(data['subject_age'].isnull().sum())

""" Using linear regression, we want to predict the age of the person who has been stopped"""

# Let us first define the categorical variables

not_categorical_vars = ['raw_row_number',
                        'date',
                        'subject_age']

for categorical in list(data.columns):
    if categorical not in not_categorical_vars:
        data[categorical] = data[categorical].astype('category')

print(data.info())
# let us now encode the data. As we have a lot of categories among the variables 

categorical_vars = ['subject_sex',
                    'subject_race',
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
                    'vehicle_registration_state',
                    'raw_Race']


def make_dummies(dataset, dummy_list):
    for i in dummy_list:
        dummy = pd.get_dummies(dataset[i], prefix= i, dummy_na= False)
        dataset = dataset.drop(i,1)
        dataset = pd.concat([dataset,dummy], axis = 1)
    return dataset

new_data =make_dummies(data,categorical_vars)

print(new_data.head)

# Simple linear regression to predict the age of the person stopped by the police 


class LinearRegression():
    """
    predicts the age of person stopped taking input of 
    the history of the stops
    """

    def __init__(self, learning_rate = 0.0001, epochs = 100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bais = None

    def fit(self,X,y):
        """ Takes in the training samples and labels"""
        # initialize the parameters 
        number_samples , number_features = X.shape
        self.weights = np.zeros(number_features)
        self.bais = 0
        
        for _ in range(self.epochs):
            predicted_y = np.dot(X, self.weights) + self.bias
            #derivative of weight
            dw =(1/number_samples) * np.dot(X.T , (predicted_y - y))
            # Derivative of bias 
            db = (1/ number_samples) * np.sum(predicted_y - y)

            self.weights -= self.learning_rate * dw
            self.bais -= self.learning_rate * db
    

    def pred(self, X):
        predicted_y = np.dot(X, self.weights) + self.bias
        return predicted_y



