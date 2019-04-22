##########
# This file contains utility methods for handling the data
##########

##########
# Imports
##########
import pandas as pd
from sklearn.model_selection import train_test_split
import os

# getStartData()
# Gets a bootstrapped sample of the data left over from the safe split
# Requires the normalizeddata_train.csv and normalizeddata_train_countries.csv files
# to be in the data folder
#
# PARAM boolean countries   :   True if you want the region as a predictor,
#                               False if you want region as a response
#                               defaults to True
# RETURN pandas.DataFrame   :   Dataframe of ~12,000 bootstrapped points from the dataset
def getStartData(path='data',countries=True):
    if countries:
        df = pd.read_csv(os.path.join(path,'normalizeddata_train_countries.csv'))
    else:
        df = pd.read_csv(os.path.join(path,'normalizeddata_train.csv'))

    df = bootstrap(df)
    return df


# splitData()
# Randomly splits a dataframe in two
#
# PARAM pandas.DataFrame df :   The dataframe to split
#       int/float size      :   Number of rows in the first dataframe returned if int
#                               Percent of rows in the first dataframe returned if float
# RETURN (pandas.DataFrame, pandas.DataFrame)
#                           :   (dataframe of the size given by size, remainder of the
#                                original dataframe)
def splitData(df, size):
    return train_test_split(df, train_size=size)


# bootstrap()
# Gets a bootstrapped sample of a given size from the dataframe
#
# PARAM pandas.DataFrame df :   The dataframe to pull the sample from
# PARAM int/float size      :   Number of rows in sample if int
#                               Percent of original df if float
#                               defaults to 1.0 (aka size of full dataframe)
# RETURN pandas.DataFrame   :   Bootstrapped sample
def bootstrap(df, size=1.0):
    if isinstance(size, int):
        return df.sample(n=size, replace=True)
    else:
        return df.sample(frac=size, replace=True)

# Takes an int for rank
def setDataset(df, rank):
    df.loc[df['Position'] <= rank, 'Position'] = rank
    df.loc[df['Position'] > rank, 'Position'] = 200
    return df
