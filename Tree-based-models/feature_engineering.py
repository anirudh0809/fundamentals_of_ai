import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 
from sklearn.base import TransformerMixin

data = pd.read_csv('/Users/anirudhsharma/FAI/ma_statewide_2020_04_01.csv', low_memory = False)

''' handling all the categorical features and handling all the null values and missing values'''

print(f'Shape of the data -',data.shape)
print('\n')
print(f'Glimpse of the data',data.head(10))
print('\n')
print(f'Some information about the data -', data.info())
print('\n')
print('\n')
print('\n')
print('\n')
print('\n')
print('\n')
print(f'Number of null values in the data', data.isnull().sum())

# handling the missing values in subject age column with median of the values in that column
mean1 = data['subject_age'].mean()
data['subject_age'].fillna(mean1, inplace = True)
print('\n')
print('\n')
print('\n')
print('\n')
print('\n')
print('\n')
print(f'Null values in subject_age -',data['subject_age'].isnull().sum())

# handling missing values in 

class DataFrameImputer(TransformerMixin):

    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

data_new = pd.DataFrame(data)
data_new = DataFrameImputer().fit_transform(data_new)

print('\n')
print('Before')
print(data)
print('\n')
print('After')
print('\n')
print(data_new)
print('\n')
print('\n New Null values in the columns \n' , data_new.isnull().sum())

#Let us first define the categorical variables

not_categorical_vars = ['raw_row_number',
                        'date',
                        'subject_age']

for categorical in list(data_new.columns):
    if categorical not in not_categorical_vars:
        data_new[categorical] = data_new[categorical].astype('category')

print(data_new.info())

# adding new date month and year column to the data and removing redundant columns
data_new['date'] = pd.to_datetime(data_new['date'])
data_new['year'] = data_new['date'].dt.year
data_new['month'] = data_new['date'].dt.month
data_new['day'] = data_new['date'].dt.day

columns_to_be_removed = ['location',
                         'county_name',
                         'raw_Race',
                         'raw_row_number' ]

data_new = data_new.drop(columns_to_be_removed,axis = 1)

print(list(data_new.columns))











