import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from dataUtilities import splitData
import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection as ms
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
from sklearn import metrics



df = pd.read_csv('data/normalizeddata_train.csv')
#train, test = splitData(df, 0.8);
#df.shape
#train.shape
#test.shape
#X_train = train.drop(['Region', 'URL', 'Position'], inplace=False, axis=1)
# print(X.dtypes)
#y_train = train[['Position']]

X = df.drop(['Region', 'URL', 'Position'], inplace=False, axis=1)
y = df[['Position']]

from sklearn.neighbors import KNeighborsClassifier
# neigh = KNeighborsClassifier(n_neighbors=k)
# neigh.fit(X_train, y_train)
# print(neigh.predict([X_train.iloc[50]]))
# print(neigh.predict_proba([X_train.iloc[50]]))
#
#
#X_test = test.drop(['Region', 'URL', 'Position'], inplace=False, axis=1)
# # print(X.dtypes)
#y_test = test[['Position']]
# print(neigh.score(X_test, y_test, None)) # Returns the mean accuracy on the given test data and labels.

k_range = range(1, 100)
scores = {}
scores_list = []

for k in k_range:
    neigh = KNeighborsClassifier(n_neighbors=k)
    scores[k] = ms.cross_val_score(neigh, X, y, cv=4)
    #neigh.fit(X_train, y_train)
    #y_pred = neigh.predict(X_test)
    #print(neigh.predict_proba([X_train.iloc[50]]))
    #scores[k] = metrics.accuracy_score(y_test,y_pred)
    print(k)
    print(scores[k])
    meanScore = np.mean(scores[k])
    print(meanScore)
    print('-----')
    scores_list.append(meanScore)

plt.plot(k_range, scores_list)
#import scikitplot as skplt
#skplt.metrics.plot_precision_recall_curve(k_range, scores_list)
plt.show()

bestK = scores_list.index(max(scores_list)) + 1;

# # create the plot
# h = .02  # step size in the mesh
#
# # Create color maps
# cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
# cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
#
# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, x_max]x[y_min, y_max].
# x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
# y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
#                      np.arange(y_min, y_max, h))
# Z = neigh.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure()
# plt.pcolormesh(xx, yy, Z, cmap=cmap_light)
#
# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
#             edgecolor='k', s=20)
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.title("3-Class classification (k = %i, weights = '%s')"
#           % (n_neighbors, weights))