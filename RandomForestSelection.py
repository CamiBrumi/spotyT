import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

#reading csv file into dataframe
df = pd.read_csv('engineered-track-data.csv')
df = df.copy()
df = df.dropna()

#implementing lasso selection
#dropping fields that will not count as predictors
df.drop(['artist_name', 'track_name', 'track_id'], inplace=True, axis=1)

#normalizing dataframe values
#returns a numpy array (headers not included)
df_array = df.values
scaled_data = preprocessing.scale(df)

#assigning response variable
y = scaled_data[:,0]

#assigning predictors
x = scaled_data[:,1:]

#create a random forest classifier
rf = RandomForestRegressor()

#train the classifier
rf.fit(x, y)

#reassigning column names to normalized matrix
df_scaled = pd.DataFrame(scaled_data, columns=df.columns)

# Print the name and variance decrease (importance measure) of each feature
print (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), df_scaled.columns.values[1:]),
             reverse=True))