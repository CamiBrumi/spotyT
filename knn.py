import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv('data/normalizeddata_train.csv')
X = df.drop(['Region', 'URL', 'Position'], inplace=False, axis=1)
print(X.dtypes)
y = df[['Position']]
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=100)
neigh.fit(X, y)
print(neigh.predict([X.iloc[50]]))
print(neigh.predict_proba([X.iloc[50]]))

