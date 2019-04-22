##########
# This file contains utility methods for handling the data
##########

##########
# Imports
##########
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn_pandas import DataFrameMapper
from sklearn import preprocessing
import os
from sklearn.utils import resample
import math

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

""" takes in name of csv file and creates dataframe of file
@:param file_name: str 
@:return df
@:return df_x
@:return df_y
returns dataframe with elements of csv file, predictor set, and label set
"""
def prepare_df(position):
    df = getStartData('../data', True)

    df = setDataset(df, position)
    df, df_test = splitData(df, 0.8)

    max_size = df['Position'].value_counts().iloc[0]
    max_index = df['Position'].value_counts().index[0]
    df_temp = df[(df['Position'] == position)]
    df_200 = df[(df['Position'] == 200)]

    if (max_index == 200):
        df_temp = resample(df_temp, replace=True, n_samples=max_size)
    else:
        df_200 = resample(df_200, replace=True, n_samples=max_size)

    df = pd.concat([df_temp, df_200])
    print(df['Position'].value_counts())

    df_y = df['Position']
    df_x = df.drop(['Position', 'URL'], axis=1)
    df_test_y = df_test['Position']
    df_test_x = df_test.drop(['Position', 'URL'], axis=1)
    return df_x, df_y, df_test_x, df_test_y

def normalize(df):
    mapper = DataFrameMapper(
        [('Position', None)], input_df=True, default=preprocessing.MinMaxScaler())
    df = pd.DataFrame(mapper.fit_transform(df))
    return df

# df is test dataframe, models is a list of models
# def layeredBinaryClassification(df, models):
#     pred = pd.DataFrame()
#     for m in models:
#         p = m.predict(df)

