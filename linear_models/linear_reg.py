import pandas as pd 
import numpy as np 
from sklearn.linear_model import LinearRegression
from sklearn.base import TransformerMixin
import datetime

data = pd.read_csv('/Users/anirudhsharma/FAI/ma_statewide_2020_04_01.csv', low_memory = False)
print(data.head(10))

# Checking all the nan values and handling them
print(data.isnull().sum())


df = pd.DataFrame(data)
median1 = df['subject_age'].median()
df['subject_age'].fillna(median1, inplace = True)

#Segregate the values based on the categories, remove the nulls and normalize the data column
df['race'] = pd.Series(len(df['subject_race']), index=df.index)
df['race'] = 0

#To assign null values
df.loc[(df['subject_race'] != 'hispanic') | 
           (df['subject_race'] != 'white') |
           (df['subject_race'] != 'black') |
           (df['subject_race'] != 'asian/pacific islander') |
           (df['subject_race'] != 'other') |
           (df['subject_race'].isnull() == True), 'race'] = np.nan

#To assign the categorical values to the dataframe 'race'
df.loc[(df['subject_race'] == 'hispanic') | 
           (df['subject_race'] == 'white') |
           (df['subject_race'] == 'black') |
           (df['subject_race'] == 'other') |
           (df['subject_race'] == 'asian/pacific islander'), 'race'] = df['subject_race']

race_copy = df['race'].copy(deep = True)

# Fill NaN values.
df['race'].fillna(value = 1, inplace = True)

# Obtain values for every race.Axis=0 for rows
race_copy.dropna(axis = 0, inplace = True)
sorted_race = race_copy.value_counts(normalize = True).sort_index()

# Fill one values for individual person with randomly picked from random choice.
df['race'] = df['race'].apply(lambda x: np.random.choice([x for x in sorted_race.index],
                                replace = True, p = sorted_race) if (x == 1) else x).astype(str)

#Normalize=True prints the relative frequency of the values
print("\nFilled NaNs normalized:\n", df['race'].value_counts(normalize = True))

df['subject_race'] = df['race']
df['subject_race'].value_counts()

#Segregate the values based on the categories, remove the nulls and normalize the data column
df['sex'] = pd.Series(len(df['subject_sex']), index = df.index)
df['sex'] = 0

# Randomly stick sex to every user with NaN value.
df.loc[(df['subject_sex'] != 'male') | 
           (df['subject_sex'] != 'female') |
           (df['subject_sex'].isnull() == True), 'sex'] = np.nan
df.loc[(df['subject_sex'] == 'male') | 
           (df['subject_sex'] == 'female'), 'sex'] = df['subject_sex']


# Create a copy to calculate proportions.
sex_copy = df['sex'].copy(deep = True)

# Fill NaN values.
df['sex'].fillna(value = 1, inplace = True)

# Obtain values for every sex.
sex_copy.dropna(axis = 0, inplace = True)
sorted_sex = sex_copy.value_counts(normalize = True).sort_index()

# Fill one values in suspector_sex_rand with randomly picked from random choice.
df['sex'] = df['sex'].apply(lambda x: np.random.choice([x for x in sorted_sex.index],
                                replace = True, p = sorted_sex) if (x == 1) else x).astype(str)
print("Gender proportions after filled NaNs: \n", df['sex'].value_counts(normalize = True))

df['subject_sex'] = df['sex']
df['subject_sex'].value_counts()


#Segregate the values based on the categories, remove the nulls and normalize the data column
df['outcome_v'] = pd.Series(len(df['outcome']), index = df.index)
df['outcome_v'] = 0

# Randomly stick sex to every user with NaN value.
df.loc[(df['outcome'] != 'citation') | 
           (df['outcome'] != 'warning') |
           (df['outcome'] != 'arrest') |
           (df['outcome'].isnull() == True), 'outcome_v'] = np.nan
df.loc[(df['outcome'] != 'citation') | 
           (df['outcome'] != 'warning') |
           (df['outcome'] != 'arrest'), 'outcome_v'] = df['outcome']


# Create a copy to calculate proportions.
outcome_copy = df['outcome_v'].copy(deep = True)

# Fill NaN values.
df['outcome_v'].fillna(value = 1, inplace = True)

outcome_copy.dropna(axis = 0, inplace = True)
sorted_outcome = outcome_copy.value_counts(normalize = True).sort_index()

# Fill one values in suspector_sex_rand with randomly picked from random choice.
df['outcome_v'] = df['outcome_v'].apply(lambda x: np.random.choice([x for x in sorted_outcome.index],
                                replace = True, p = sorted_outcome) if (x == 1) else x).astype(str)
print("Outcome proportions after filled NaNs: \n", df['outcome_v'].value_counts(normalize = True))

df['outcome'] = df['outcome_v']
df['outcome'].value_counts()

#Segregate the values based on the categories, remove the nulls and normalize the data column
df['vehicle'] = pd.Series(len(df['vehicle_type']), index = df.index)
df['vehicle'] = 0

df.loc[(df['vehicle_type'] != 'Commerical') | 
           (df['vehicle_type'] != 'Passenger') |
           (df['vehicle_type'] != 'Motorcycle') |
           (df['vehicle_type'] != 'Taxi/Livery') |
           (df['vehicle_type'] != 'Trailer') |
           (df['vehicle_type'].isnull() == True), 'vehicle'] = np.nan
df.loc[(df['vehicle_type'] != 'Commerical') | 
           (df['vehicle_type'] != 'Passenger') |
           (df['vehicle_type'] != 'Motorcycle') |
           (df['vehicle_type'] != 'Taxi/Livery') |
           (df['vehicle_type'] != 'Trailer'), 'vehicle'] = df['vehicle_type']


# Create a copy to calculate proportions.
outcome_copy = df['vehicle'].copy(deep = True)

# Fill NaN values.
df['vehicle'].fillna(value = 1, inplace = True)

outcome_copy.dropna(axis = 0, inplace = True)
sorted_outcome = outcome_copy.value_counts(normalize = True).sort_index()

# Fill one values in suspector_sex_rand with randomly picked from random choice.
df['vehicle'] = df['vehicle'].apply(lambda x: np.random.choice([x for x in sorted_outcome.index],
                                replace = True, p = sorted_outcome) if (x == 1) else x).astype(str)
print("Vehicle Type proportions after filled NaNs: \n", df['vehicle'].value_counts(normalize = True))

df['vehicle_type'] = df['vehicle']
df['vehicle_type'].value_counts()

print(df.isnull().sum())

print(df['arrest_made'].unique())
print(df['citation_issued'].unique())
print(df['warning_issued'].unique())

# Convert the date into segments for day , date and year 

df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day

print(df['month'].head)


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

X = pd.DataFrame(df)
new_data = DataFrameImputer().fit_transform(X)

print('before...')
print(X)
print('after...')
print(new_data)

print(new_data.isnull().sum())



""" Using linear regression, we want to predict the age of the person who has been stopped"""

#Let us first define the categorical variables

not_categorical_vars = ['raw_row_number',
                        'date',
                        'subject_age']

for categorical in list(new_data.columns):
    if categorical not in not_categorical_vars:
        new_data[categorical] = new_data[categorical].astype('category')

print(new_data.info())

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
                    'raw_Race',
                    'race',
                    'sex',
                    'outcome_v',
                    'vehicle']


def make_dummies(dataset, dummy_list):
    for i in dummy_list:
        dummy = pd.get_dummies(dataset[i], prefix= i, dummy_na= False)
        dataset = dataset.drop(i,1)
        dataset = pd.concat([dataset,dummy], axis = 1)
    return dataset

dummy_data =make_dummies(new_data,categorical_vars)

print(dummy_data.head)
print(dummy_data['subject_age'].head)

# Simple linear regression using sklearn to predict the age of the person stopped 
""" Define dependent and independent variables"""

X = dummy_data.drop(['raw_row_number','subject_age','date','location','county_name'], axis= 1)
#X = X.values.reshape(-1,1)

Y = dummy_data['subject_age']
#
""" First we need to split the dataset in to test and train and dependent and independent variables"""
print("Before splitting")
from sklearn.model_selection import train_test_split
X_train, X_test, y_train , y_test = train_test_split(X,Y, test_size= 0.25, random_state = 12345)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


print("Training start")
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)
print("Trained")

lm_pred = model.predict(X_test)

from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, mean_squared_error, mean_absolute_error, explained_variance_score

mse = mean_squared_error(y_test, lm_pred)
mae = mean_absolute_error(y_test, lm_pred)
evs = explained_variance_score(y_test, lm_pred)

metrics = {"mse":mse,
           "mae":mae,
           "evs":evs
}

print(metrics)

# import statsmodels.api as sm

# X_train_new = sm.add_constant(X_train)
# lm_1 = sm.OLS(y_train, X_train).fit()
# print(lm_1.summary())


# Simple linear regression from scratch to predict the age of the person stopped by the police 


class lin_reg():
    """
    predicts the age of person stopped taking input of 
    the history of the stops
    """

    def __init__(self, learning_rate = 0.001, epochs = 100):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None

    def fit(self,X,y):
        """ Takes in the training samples and labels"""
        # initialize the parameters 
        number_samples , number_features = X.shape
        self.weights = np.zeros(number_features)
        self.bias = 0
        
        for _ in range(self.epochs):
            predicted_y = np.dot(X, self.weights) + self.bias
            #derivative of weight
            dw =(1/number_samples) * np.dot(X.T , (predicted_y - y))
            # Derivative of bias 
            db = (1/ number_samples) * np.sum(predicted_y - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    

    def pred(self, X):
        predicted_y = np.dot(X, self.weights) + self.bias
        return predicted_y



regressor = lin_reg()
regressor.fit(X_train, y_train)
predicted_vals = regressor.predict(X_test)

def mean_sq(y_true, y_predicted):
    np.mean((y_true - y_predicted)** 2)

mse_value = mean_sq(y_test,predicted_vals)
print(mse_value)    