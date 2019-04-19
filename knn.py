import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier

# knn()
# Chooses the K for the Knn model that gives the best accuracy.
#
# RETURN best k
#
def knn():
    df = pd.read_csv('data/normalizeddata_train.csv')
    X = df.drop(['Region', 'URL', 'Position'], inplace=False, axis=1)
    y = df[['Position']]

    k_range = range(1, 100)
    scores = {}
    scores_list = []

    for k in k_range:
        neigh = KNeighborsClassifier(n_neighbors=k)
        scores[k] = ms.cross_val_score(neigh, X, y, cv=4)
        meanScore = np.mean(scores[k])
        scores_list.append(meanScore)

    plt.plot(k_range, scores_list)
    plt.show()

    return scores_list.index(max(scores_list)) + 1;

