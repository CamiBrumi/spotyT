import itertools
import time
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np


#----------- DIFFERENT TYPES OF SUBSET SELECTION ALGORITHMS -----------

spotify_df = pd.read_csv('SpotifyFeatures.csv')
spotify_df.head()

# generate dummy variables for categorical data
dummies = []
columns = ['genre', 'key', 'mode', 'time_signature']
for name in columns:
    dummies.append(pd.get_dummies(spotify_df[name]))

y = spotify_df["popularity"]


# Drop the column with the independent variable (Salary), and columns for which we created dummy variables
X_ = spotify_df.drop(['popularity', 'track_id', 'track_name', 'genre', 'artist_name', 'key', 'mode', 'time_signature'], axis=1).astype('float64')
#print(X_)
# Define the feature set X.
X = X_
for element in dummies:
    X = pd.concat([X, element], axis = 1)
#print number of rows and columns in feature set
print("dims of feature set X is: ", len(X))

# implementing best subset selection

# helper method
def processSubset(feature_set):
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(y, X[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X[list(feature_set)]) - y) ** 2).sum()
    return {"model": regr, "RSS": RSS}


# selection algorithm
# returns DataFrame containing the best model that was generated,
# and some extra information about the model
def getBest(k):
    tic = time.time

    results = []

    for combo in itertools.combinations(X.columns, k):
        results.append(processSubset(combo))

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)

    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]

    toc = time.time
    print("Processed", models.shape[0], "models on", k, "predictors in", (toc - tic), "seconds.")

    # Return the best model, along with some other useful information about the model
    return best_model


# calling the above function for each predictor k
# takes a while to complete
models_best = pd.DataFrame(columns=["RSS", "model"])

tic = time.time
for i in range(1, 8):
    models_best.loc[i] = getBest(i)

toc = time.time
print("Total elapsed time:", (toc - tic), "seconds.")

#this dataframe contains the best models we have generated, and their RSS
models_best

#showing details for each subset
for x in range(X.num_columns):
    print((models_best).loc[x,"model"].summary())

#plotting RSS, R**2, AIC, and BIC  for all models at once
#helps us decide which model to select
#type=="l" option tells R to connect the plotted points with lines
#will decide which predictors are better when we decide what kind of error
#calculation method we will be using in the rest of the project
plt.figure(figsize=(20,10))
plt.rcParams.update({'font.size': 18, 'lines.markersize': 10})

# Set up a 2x2 grid so we can look at 4 plots at once
plt.subplot(2, 2, 1)

# plot a red dot to indicate the model with the largest adjusted R^2 statistic.
# argmax() function can be used to identify the location of the maximum point of a vector
plt.plot(models_best["RSS"])
plt.xlabel('# Predictors')
plt.ylabel('RSS')

# plot a red dot to indicate the model with the largest adjusted R^2 statistic.
# argmax() function can be used to identify the location of the maximum point of a vector

rsquared_adj = models_best.apply(lambda row: row[1].rsquared_adj, axis=1)

plt.subplot(2, 2, 2)
plt.plot(rsquared_adj)
plt.plot(rsquared_adj.argmax(), rsquared_adj.max(), "or")
plt.xlabel('# Predictors')
plt.ylabel('adjusted rsquared')

# do the same for AIC and BIC, this time looking for the models with the SMALLEST statistic
aic = models_best.apply(lambda row: row[1].aic, axis=1)

plt.subplot(2, 2, 3)
plt.plot(aic)
plt.plot(aic.argmin(), aic.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('AIC')

bic = models_best.apply(lambda row: row[1].bic, axis=1)

plt.subplot(2, 2, 4)
plt.plot(bic)
plt.plot(bic.argmin(), bic.min(), "or")
plt.xlabel('# Predictors')
plt.ylabel('BIC')

#--------------------- BEST SUBSET IMPLEMENTATION ENDS HERE ---------------

#implementing forwards and backwards stepwise selection
def forward(predictors):
    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X.columns if p not in predictors]

    tic = time.time()

    results = []

    for p in remaining_predictors:
        results.append(processSubset(predictors + [p]))

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)

    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]

    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors) + 1, "predictors in", (toc - tic), "seconds.")

    # Return the best model, along with some other useful information about the model
    return best_model

# calling the above function for each predictor k
# runs much faster than best subset
models_fwd = pd.DataFrame(columns=["RSS", "model"])

tic = time.time()
predictors = []

for i in range(1,len(X.columns)+1):
    models_fwd.loc[i] = forward(predictors)
    predictors = models_fwd.loc[i]["model"].model.exog_names

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")

#showing details for each subset
for x in range(X.num_columns):
    print(models_fwd.loc[x,"model"].summary())

#---------------------- FORWARD SELECTION COMPLETE -------------------

#implementing Backward selection
#only difference is looping though predictors in reverse
#showing details for each model
def backward(predictors):
    tic = time.time()

    results = []

    for combo in itertools.combinations(predictors, len(predictors) - 1):
        results.append(processSubset(combo))

    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)

    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]

    toc = time.time()
    print("Processed ", models.shape[0], "models on", len(predictors) - 1, "predictors in", (toc - tic), "seconds.")

    # Return the best model, along with some other useful information about the model
    return best_model

#calling above function for each predictor k
models_bwd = pd.DataFrame(columns=["RSS", "model"], index = range(1,len(X.columns)))

tic = time.time()
predictors = X.columns

while(len(predictors) > 1):
    models_bwd.loc[len(predictors)-1] = backward(predictors)
    predictors = models_bwd.loc[len(predictors)-1]["model"].model.exog_names

toc = time.time()
print("Total elapsed time:", (toc-tic), "seconds.")

#comparing between models

print("------------")
print("Best Subset:")
print("------------")
for x in range(X.num_columns):
    print((models_best).loc[x,"model"].summary())


print("------------")
print("Forward Selection:")
print("------------")
for x in range(X.num_columns):
    print((models_fwd).loc[x,"model"].summary())

print("------------")
print("Forward Selection:")
print("------------")
for x in range(X.num_columns):
    print((models_fwd).loc[x, "model"].summary())



