import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

# knn()
# Chooses the K for the Knn model that gives the best accuracy.
# PARAM mink
# RETURN best k
#
from NewModeling.RandomForestWithKFold import plot_confusion_matrix, prepare_df_wo_countries


def knn(mink = 1, maxk = 100, rank=10):

    prepare_df_wo_countries('NewModeling/normalizeddata_train_countries.csv', rank)
    df = pd.read_csv('NewModeling/normalizeddata_train.csv')
    X = df.drop(['Region', 'URL', 'Position'], inplace=False, axis=1)
    y = df[['Position']]

    k_range = range(mink, maxk)
    scores = {}
    scores_list = []


    for k in k_range:
        neigh = KNeighborsClassifier(n_neighbors=k)
        scores[k] = ms.cross_val_score(neigh, X, y, cv=4)
        meanScore = np.mean(scores[k])
        scores_list.append(meanScore)
        print(k)

    plt.plot(k_range, scores_list)
    plt.show()

    max_score = max(scores_list)
    print("max score is:", max_score)
    max_k = scores_list.index(max(scores_list))
    return max_k




def knn_score(k_val, rank):
    df, df_test_x, df_test_y, df_train_x, df_train_y = prepare_df_wo_countries('NewModeling/normalizeddata_train_countries.csv',rank)

    # df = pd.read_csv('data/normalizeddata_train.csv') #change to test later
    # X = df.drop(['Region', 'URL', 'Position'], inplace=False, axis=1)
    # y = df[['Position']]
    neigh = KNeighborsClassifier(n_neighbors=k_val)
    score = ms.cross_val_score(neigh, df_train_x, df_train_y, cv=4)

    #plots and matrices

    # checking model with test set
    clf = KNeighborsClassifier(n_neighbors=k_val)
    clf.fit(df_train_x, df_train_y)
    predictions = clf.predict(df_test_x)
    print("Test Accuracy for ideal parameters:", metrics.accuracy_score(predictions, df_test_y))

    # getting confusion matrix
    pd.np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix
    plot_confusion_matrix(df, df_test_y, predictions, title='Confusion matrix, without normalization')

    # Plot normalized confusion matrix
    plot_confusion_matrix(df, df_test_y, predictions, normalize=True, title='Normalized confusion matrix')

    print(score)
    return score

#----------------------- RUNNING CODE HERE -----------------------------------
knn_score(9,10)

#
# k = knn(1, 15)
# print('best k: ')
# print(k)

