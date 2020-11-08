import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import time 

data = pd.read_csv('C:/Users/SwetaMankala/Desktop/Assignments/EAI6000/ma_statewide_2020_04_01.csv',low_memory= False)
data.head(10)
# Checking the shape of the data set 
data.shape
data['location'].unique()
# Using the commqand below we can see what exactly our columns look like 
data.columns
data['county_name'].unique()
data['subject_race'].unique()
data.info()
n=data['raw_row_number'].count()
x=data['subject_race'].value_counts()
print('Numerical columns:',data.select_dtypes(include=np.number).columns)
print('Categorical columns:',data.select_dtypes(include='object').columns)


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

#Convert the object type variables to string
df['subject_sex'] = df['subject_sex'].astype(str)
df['subject_race'] = df['subject_race'].astype(str)
df['type'] = df['type'].astype(str)
df['arrest_made'] = df['arrest_made'].astype(str)
df['citation_issued'] = df['citation_issued'].astype(str)
df['outcome'] = df['outcome'].astype(str)
df['contraband_found'] = df['contraband_found'].astype(str)
df['contraband_drugs'] = df['contraband_drugs'].astype(str)
df['warning_issued'] = df['warning_issued'].astype(str)
df['contraband_weapons'] = df['contraband_weapons'].astype(str)
df['contraband_alcohol'] = df['contraband_alcohol'].astype(str)
df['contraband_other'] = df['contraband_other'].astype(str)
df['frisk_performed'] = df['frisk_performed'].astype(str)
df['search_conducted'] = df['search_conducted'].astype(str)
df['search_basis'] = df['search_basis'].astype(str)
df['reason_for_stop'] = df['reason_for_stop'].astype(str)
df['vehicle_type'] = df['vehicle_type'].astype(str)
df['vehicle_registration_state'] = df['vehicle_registration_state'].astype(str)
df['raw_Race'] = df['raw_Race'].astype(str)

data[data.subject_sex == "male"].location.value_counts()
data[data.subject_sex == "female"].location.value_counts()

# If we want to see number of violations per gender with respect to their race

race = data.groupby(["subject_sex"]).subject_race.value_counts(normalize= True).unstack()
race

plt.figure(figsize=(12, 8))
race.black.plot(kind="bar")

plt.figure(figsize=(12, 8))
race.white.plot(kind="bar")

plt.figure(figsize=(12, 8))
race.hispanic.plot(kind="bar")

# We want to check which year had the least number of stops 

data.date
data['date'] = pd.to_datetime(data.date, format="%Y-%M-%d")
data["year"] = data.date.dt.year

import math
import seaborn as sns
sns.set_style('whitegrid')

# Rounding the integer to the next hundredth value plus an offset of 100
def round(x):
    return 100 + int(math.ceil(x / 100.0)) * 100 
sns.catplot("subject_sex", col = "reason_for_stop", col_wrap = 3,data = data[data.reason_for_stop.notnull()],kind = "count")
# Get current axis on current figure
axis = plt.gca()

# ylim max value to be set
max_y = data['subject_sex'].value_counts().max() 
axis.set_ylim([0, round(max_y)])

# Iterate through the list of axes' patches
for p in axis.patches:
    axis.text(p.get_x() + p.get_width()/2., p.get_height(), '%d' % int(p.get_height()), 
            fontsize=12, color='red', ha='center', va='bottom')
plt.show()

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

df['subject_race'] = label_encoder.fit_transform(df['subject_race'])
df['arrest_made'] = label_encoder.fit_transform(df['arrest_made'])
df['citation_issued'] = label_encoder.fit_transform(df['citation_issued'])
df['outcome'] = label_encoder.fit_transform(df['outcome'])
df['contraband_found'] = label_encoder.fit_transform(df['contraband_found'])
df['contraband_drugs'] = label_encoder.fit_transform(df['contraband_drugs'])
df['contraband_weapons'] = label_encoder.fit_transform(df['contraband_weapons'])
df['contraband_alcohol'] = label_encoder.fit_transform(df['contraband_alcohol'])
df['contraband_other'] = label_encoder.fit_transform(df['contraband_other'])

corr = df.corr()
corr.style.background_gradient(cmap='coolwarm')

