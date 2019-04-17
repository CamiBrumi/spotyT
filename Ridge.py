#Ridge Regression

#get_ipython().magic('matplotlib inline')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"C:\Users\Yanni Peri-Okonny\Documents\R\Project\normalizeddata.csv").dropna().drop('1', axis=1).drop('2', axis=1)
#df.info()
#df.columns

y = df['0']
#print(y)
X_ = df.drop('0', axis=1).astype('float64')
X = pd.concat([X_], axis=1)
#X.info()

alphas = 10 ** np.linspace(10, -2, 100) * 0.5
#print("Alphas: ",alphas)

ridge = Ridge(normalize=True)
coefs = []

for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

np.shape(coefs)

ax = plt.gca()
ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.axis('tight')
plt.xlabel('alpha')
plt.ylabel('weights')

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

#RIDGECV (Ridge with Cross Validation)

ridgecv = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', normalize=True)
ridgecv.fit(X_train, y_train)
print("RidgeCV Alpha: ", ridgecv.alpha_)
#
ridge4 = Ridge(alpha=ridgecv.alpha_, normalize=True)
ridge4.fit(X_train, y_train)
print("Test MSE: ", mean_squared_error(y_test, ridge4.predict(X_test)))

ridge4.fit(X, y)
pd.Series(ridge4.coef_, index=X.columns)
print("Coefficients: ")
print(pd.Series(ridge4.coef_, index=X.columns))  # Print coefficients