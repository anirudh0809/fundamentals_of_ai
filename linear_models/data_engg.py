import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import os 

data = pd.read_csv('/Users/anirudhsharma/FAI/ma_statewide_2020_04_01.csv', low_memory = False)
print(data.head(10))

def feature_engg(dataset):
    """
    docstring
    """
    df = pd.DataFrame(dataset)
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
    ex_copy.dropna(axis = 0, inplace = True)
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

    #  Fill one values in suspector_sex_rand with randomly picked from random choice.
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
    