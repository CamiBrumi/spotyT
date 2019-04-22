# import matplotlib.pyplot as plt
# from mpl_toolkits import mplot3d
from sklearn import svm, metrics
from sklearn.model_selection import KFold, cross_val_score
from sklearn.utils import resample
import pandas as pd
import math
from dataUtilities import *

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
def make_kfold_cvset(k, df_x):
    kf = KFold(n_splits=k, shuffle=True)  # defines the k fold split, with shuffling
    return kf.split(df_x)

    # # getting the folds as a result - can call in other functions
    # for train_idx, test_idx in kf.split(df_x):
    #     x_train, x_test = df_x[train_idx], df_x[test_idx]
    #     y_train, y_test = df_y[train_idx], df_y[test_idx]


"""runs multi-class SVM for a certain set of parameters,
then cross validates it using k-fold cv with a certain number of folds
@:param C : int 
@:param kernel : 'linear', 'rbf', 'sigmoid', 'poly'
@:param degree : int (only used with poly)
@:param shrinking : boolean
@:param n_folds : int
@:param df_x : dataFrame
@:param df_y : dataFrame
returns array of scores, with no of entries = no of folds
"""
def svm_accuracy(C, kernel, degree, n_folds, df_x, df_y):
    # creating instance of random forest classifier
    if degree == 1:
        clf = svm.SVC(C=C, kernel=kernel, gamma='auto')
    else:
        clf = svm.SVC(C=C, kernel=kernel, gamma='auto', degree=degree)

    # calculating accuracy of model
    # cv value is the number of folds for cross validation
    # will train and test for you, then show you accuracy for each fold
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
def find_highest_accuracy (c_min, c_max, deg_min, deg_max, n_folds_min, n_folds_max):

    all_data_list = []
    final_arr = []

    # getting dataframe from filename
    df_x, df_y, df_test_x, df_test_y = prepare_df(100)


    # # iterating through parameters to get scores
    # for n_folds in range (n_folds_min, n_folds_max+1):
    #     for k in range(4):
    #         kSwitch = {
    #             0: 'linear',
    #             1: 'rbf',
    #             2: 'poly',
    #             3: 'sigmoid'
    #         }
    #         for c in range(c_min, c_max+1):
    #             if k == 2:
    #                 for d in range(deg_min, deg_max+1):
    #                     print("checking for kernel", kSwitch.get(k), "and n_folds ", n_folds,
    #                           "and penalty", c, "and degree", d)
    #                     scores = svm_accuracy(math.pow(10,c), kSwitch.get(k), d, n_folds, df_x, df_y)
    #                     all_data_list.append(
    #                         [c, kSwitch.get(k), d, n_folds, scores.mean(), scores.std() * 2])
    #             else:
    #                 # i=0
    #                 print("checking for kernel", kSwitch.get(k), "and n_folds ", n_folds,
    #                       "and penalty", c)
    #                 scores = svm_accuracy(math.pow(10,c), kSwitch.get(k), 1, n_folds, df_x, df_y)
    #                 print('done')
    #                 all_data_list.append(
    #                     [c, kSwitch.get(k), 1, n_folds, scores.mean(), scores.std() * 2])
    #
    # len_arr = len(all_data_list) #length of list that has all info
    # # turn list into numpy array and reshape so each row shows one  step
    # final_arr = pd.np.asarray(all_data_list).reshape(len_arr,6)
    #
    # # printing stuff
    # # print("parameters")
    # # print(final_arr[:,[0,1,2]])
    # # print("mean and stdev of scores")
    # # print(final_arr[:,[3,4]])
    # print(final_arr)

    # getting info
    model = svm.SVC(C=100, kernel='rbf', gamma='auto')
    m2 = svm.SVC(C=1000, kernel='rbf', gamma='auto')
    print("start")
    model.fit(df_x, df_y)
    print("half")
    m2.fit(df_x, df_y)
    print("done")
    pred = model.predict(df_test_x)
    p2 = m2.predict(df_test_x)
    # print("Pred\n", pred)
    print("Metrics:", metrics.accuracy_score(df_test_y, pred))
    print("Metrics:", metrics.confusion_matrix(df_test_y, pred))

    print("Metrics:", metrics.accuracy_score(df_test_y, p2))
    print("Metrics:", metrics.confusion_matrix(df_test_y, pred))

    #TODO:  make 3D scatterplot of all data to show accuracy trends
    return final_arr


def fitModels():
    # getting dataframe from filename
    x_arr, y_arr, x_test_arr, y_test_arr = [0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0]
    x_arr[0], y_arr[0], x_test_arr[0], y_test_arr[0] = prepare_df(10)
    x_arr[1], y_arr[1], x_test_arr[1], y_test_arr[1] = prepare_df(50)
    x_arr[2], y_arr[2], x_test_arr[2], y_test_arr[2] = prepare_df(100)
    x_arr[3], y_arr[3], x_test_arr[3], y_test_arr[3] = prepare_df(150)
    x, y, x_test, y_test = prepare_df(None)

    # getting info
    bModels = []
    for i in range(4):
        bModels.append(svm.SVC(C=10000, kernel='rbf', gamma='auto'))
        bModels[i].fit(x_arr[i], y_arr[i])
        print("Fitted", i + 1, "models")
    ovrPred = layeredBinaryClassification(x_test, bModels)

    model = svm.SVC(C=10000, kernel='rbf', gamma='auto')
    model.fit(x, y)
    mcPred = model.predict(x_test)

    print("One-VS-Rest Accuracy:\n", metrics.accuracy_score(y_test, ovrPred))
    print("One-VS-Rest Confusion Matrix:\n", metrics.confusion_matrix(y_test, ovrPred))

    print("Multi-Class Accuracy:\n", metrics.accuracy_score(y_test, mcPred))
    print("Multi-Class Confusion Matrix:\n", metrics.confusion_matrix(y_test, mcPred))

### RUNNING CODE ###

# find_highest_accuracy(-5, 1, 3, 3, 6, 6)
fitModels()

# HIGHEST ACCURACY WAS WITH RBF, C=10^1
# Top 10: 0.9133266533066132,  [[0     173] [0    1823]]
# Top 50: 0.7449899799599199,  [[121   460] [49   1366]]
# Top 100: 0.657314629258517,  [[935   236] [448   377]]
# Top 150: 0.8191382765531062, [[1634    0] [361     1]]

# One-VS-Rest Accuracy:
#  0.48096192384769537
# One-VS-Rest Confusion Matrix:
#  [[119  35   4   1   0]
#  [ 62 247  70  23  16]
#  [ 69 108 251 101  41]
#  [ 37  66 104 177  73]
#  [ 34  61  71  60 166]]
# Multi-Class Accuracy:
#  0.5245490981963928
# Multi-Class Confusion Matrix:
#  [[ 94  29  21  11   4]
#  [ 24 235 100  42  17]
#  [ 21  77 338  75  59]
#  [ 31  48 112 215  51]
#  [ 12  61  92  62 165]]
