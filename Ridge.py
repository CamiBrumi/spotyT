#Ridge Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error

df = pd.read_csv(r"C:\Users\Yanni Peri-Okonny\Documents\R\Project\normalizeddata.csv").dropna().drop('1', axis=1).drop('2', axis=1)
#df.info()
#df.columns

y = df['Position']
X_ = df.drop('Position', axis=1).astype('float64')
cols = X_.columns.tolist()

i = PolynomialFeatures(degree=2, include_bias=False, interaction_only=True)
arr = i.fit_transform(X_)
# print(arr.shape)

c = []
for n in cols:
    p = n+n
    c.append(p)
# print(c)

list3 = []
for l in cols:
    for b in cols:
        if l+b and b+l in list3:
            continue
        else:
            list3.append(l+b)
# print(list3)

for z in list3:
    if z in c:
        continue
    else:
        cols.append(z)
# print(len(cols))

X = pd.DataFrame(data=arr, columns=cols)

alphas = 10 ** np.linspace(10, -2, 100) * 0.5
#print("Alphas: ",alphas)

ridge = Ridge(normalize=True)
coefs = []

for a in alphas:
    ridge.set_params(alpha=a)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

np.shape(coefs)

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

ce = pd.Series(ridge4.coef_, index=X.columns)

print("Largest: ")
print(ce.nlargest(n=50, keep="all"))
print("Smallest: ")
print(ce.nsmallest(n=50, keep="all"))

plt.plot(ce, 'go')
plt.axhline(y=0, color='r', linestyle='-')
plt.xticks([])
plt.show()
