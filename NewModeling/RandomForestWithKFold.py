import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly
import plotly.plotly as py
#plotly.tools.set_credentials_file(username='petrakumi', api_key='••••••••••')

import plotly.graph_objs as go
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
    df_x = df.drop(['Position','URL'], axis=1)
    return df, df_x, df_y



"""divides the data into k parts,
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



"""runs random forest classification for a certain no of trees,
then cross validates it using k-fold cv with a certain number of folds
@:param n_trees : int 
@:param n_folds : int
@:param df_x : dataFrame
@:param df_y : dataFrame
returns array of scores, with no of entries = no of folds
"""
def random_forest_accuracy(n_trees, n_folds, n_features, df_x, df_y):
    #creating instance of random forest classifier
    clf = RandomForestClassifier(n_estimators=n_trees, max_features=n_features)

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
def find_highest_accuracy (n_trees_min, n_trees_max, n_features_min, n_features_max, n_folds_min, n_folds_max):

    all_data_list = []

    # getting dataframe from filename
    df, df_x, df_y = prepare_df('normalizeddata_train_countries.csv')

    # iterating through number of trees and folds to get scores
    for n_trees in range (n_trees_min,n_trees_max+1):
        for n_folds in range (n_folds_min,n_folds_max+1):
            for n_features in range (n_features_min, n_features_max+1):
                print("checking for n_trees ", n_trees, " and n_folds ", n_folds, " and n_features", n_features)
                scores = random_forest_accuracy(n_trees, n_folds, n_features, df_x, df_y)

                # adding number of trees and folds to one array
                # and score to another
                all_data_list.append([n_trees, n_folds, n_features, scores.mean(), scores.std()*2])

    len_arr = len(all_data_list) #length of list that has all info
    #turn list into numpy array and reshape so each row shows one  step
    final_arr = pd.np.asarray(all_data_list).reshape(len_arr,5)

    #printing stuff
    print("trees, folds, features")
    print(final_arr[:,[0,1,2]])
    print("mean and stdev of scores")
    print(final_arr[:,[3,4]])

    #getting info
    max_acc_idx = pd.np.argmax(final_arr,axis=0)[3]
    print("max avg accuracy is ", final_arr[max_acc_idx][3])
    print("max accuracy is for tree no ", final_arr[max_acc_idx][0])
    print("max accuracy is for fold no ", final_arr[max_acc_idx][1])
    print("max accuracy is for index no ", final_arr[max_acc_idx][2])


    # 3d scatterplot of data, using plotly
    trace1 = go.Scatter3d(x=final_arr[:,0], y=final_arr[:,2], z=final_arr[:,3],
                          mode='markers', marker=dict(size=5,color=final_arr[:,2], colorscale='Viridis', opacity=0.9))

    data = [trace1]
    layout = go.Layout(margin=dict(l=0, r=0, b=0, t=0),
                       xaxis=go.layout.XAxis(
                           title=go.layout.xaxis.Title(
                               text='Number of Trees',
                               font=dict(
                                   family='Courier New, monospace',
                                   size=18
                               )
                           )
                       ),
                       yaxis=go.layout.YAxis(
                           title=go.layout.yaxis.Title(
                               text='Number of Features',
                               font=dict(
                                   family='Courier New, monospace',
                                   size=18,
                               )
                           )
                       )
                       # zaxis=go.layout.ZAxis(
                       #     title=go.layout.yaxis.Title(
                       #         text='Average Accuracy',
                       #         font=dict(
                       #             family='Courier New, monospace',
                       #             size=18,
                       #         )
                       #     )
                       # )
                    )

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.plot(fig, filename='scatter_trees_features.html')

    #below, plotting using matplotlib
    # plt.style.use("ggplot")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(final_arr[:,0], final_arr[:,1], final_arr[:,2],zdir = 'avg accuracy', c='red')
    # ax.set_xlabel('number of trees')
    # ax.set_ylabel('number of folds')
    # ax.set_zlabel('average accuracy')
    #
    # plt.show()

    return final_arr


### RUNNING CODE ###

find_highest_accuracy(80, 90, 6, 6, 6, 6)
find_highest_accuracy(80, 90, 9, 9, 6, 6)


