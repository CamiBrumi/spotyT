#code from https://mikulskibartosz.name/forward-feature-selection-in-scikit-learn-f6476e474ddd
#implementing Lasso regression

import pandas as pd
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso
import sympy

#reading csv file into dataframe
df = pd.read_csv('engineered-track-data.csv')
df = df.copy()
df = df.dropna()

#dropping fields that will not count as predictors
df.drop(['artist_name', 'track_name', 'track_id'], inplace=True, axis=1)


#normalizing dataframe values
df_array = df.values
scaled_data = preprocessing.scale(df)

#reassigning column names to normalized matrix
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)


# checking linear dependencies

#reduced_form, inds = sympy.Matrix(df_scaled[:1500]).rref()
#print("rref results:")
#print(reduced_form)
#print(inds)
#for i in inds:
#    print(df.columns.values[i])

#assigning our response variable in the dataset
y = df_scaled['popularity']


#assigning all the other fields as predictors
x = df_scaled.drop(['popularity'], axis=1)
#print(y)
#print(x)

#using the lasso function from scikit to estimate the
#most influential predictors
estimator = Lasso()
featureSelection = SelectFromModel(estimator)
featureSelection.fit(x, y)
selectedFeatures = featureSelection.transform(x)

#getting the two most influential predictors
x.columns[featureSelection.get_support()]

#print(x.columns[featureSelection.get_support()])


