import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd

""" takes in name of csv file and creates dataframe of file
@:param file_name: str 
@:return df
@:return df_x
@:return df_y
returns dataframe with elements of csv file, predictor set, and label set
"""
def prepare_df(file_name):
    df = pd.read_csv(file_name)
    df_y = df['Position']
    df_x = df.drop(['Position','URL','Region'], axis=1)
    return df, df_x, df_y

"""
divides the data into k parts,
where each part can once be used as a validation set
while the rest can be used as a training set
@:param k : int
@param df_y : dataFrame
@:param df_x : dataFrame
@:return kf
returns array of k-fold cross validation sets
"""
def make_kfold_cvset(k, df_x, df_y):
    kf = KFold(n_splits=k, shuffle=True)  # defines the k fold split, with shuffling
    return kf.split(df_x)

    # # getting the folds as a result - can call in other functions
    # for train_idx, test_idx in kf.split(df_x):
    #     x_train, x_test = df_x[train_idx], df_x[test_idx]
    #     y_train, y_test = df_y[train_idx], df_y[test_idx]

"""runs random forest classification for a certain no of trees,
then cross validates it using k-fold cv with a certain number of folds
@:param n_trees : int 
@:param n_folds : int
@:param df_x : dataFrame
@:param df_y : dataFrame
returns array of scores, with no of entries = no of folds
"""
def random_forest_accuracy(n_trees, n_folds, df_x, df_y):
    #creating instance of random forest classifier
    clf = RandomForestClassifier(n_estimators=n_trees)

    #calculating accuracy of model
    #cv value is the number of folds for cross validation
    #will train and test for you, then show you accuracy for each fold
    scores = cross_val_score(clf, df_x, df_y, cv=n_folds)
    return scores

""" computes random forest classification and gets accuracy recursively 
for the given range of trees for the forest
and folds for k-fold cv
@:param n_trees_min : int
@:param n_trees_max : int
@:param n_folds_min : int
@:param n_folds_max : int
@retun final_arr : array 
returns an array of the tree and fold combination 
together with the mean and standard deviation produced by that combination
arr[tree_no, fold_no, score.mean(), score.std()*2]
"""
def find_highest_accuracy (n_trees_min, n_trees_max, n_folds_min, n_folds_max):

    all_data_list = []

    #getting dataframe from filename
    df, df_x, df_y = prepare_df('normalizeddata_train.csv')

    #iterating through funber of trees and folds to get scores
    for n_trees in range (n_trees_min,n_trees_max+1):
        for n_folds in range (n_folds_min,n_folds_max+1):
            print("checking for n_trees ", n_trees, " and n_folds ", n_folds)
            scores = random_forest_accuracy(n_trees, n_folds, df_x, df_y)

            #adding number of trees and folds to one array
            #and score to another
            all_data_list.append([n_trees, n_folds, scores.mean(), scores.std()*2])

    len_arr = len(all_data_list) #length of list that has all info
    #turn list into numpy array and reshape so each row shows one  step
    final_arr = pd.np.asarray(all_data_list).reshape(len_arr,4)

    #printing stuff
    print("trees and folds")
    print(final_arr[0:2:1])
    print("mean and stdev of scores")
    print(final_arr[3:4:1])

    #getting info
    print("max avg accuracy is ", final_arr.max(axis=0)[2])
    print("max accuracy is for tree no ", final_arr.max(axis=0)[0])
    print("max accuracy is for fold no ", final_arr.max(axis=0)[1])

    #TODO:  make 3D scatterplot of all data to show accuracy trends
    return final_arr


### RUNNING CODE ###

find_highest_accuracy(5,7,4,8)


