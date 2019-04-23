from joblib import load
from sklearn import metrics

from NewModeling.RandomForestWithKFold import plot_confusion_matrix
from dataUtilities import prepare_df, getSafeData

import pandas as pd


def printFeatureImportance(printFeats=True, toFile=True, filename='FeatureImportances.csv', countries=True):
    clf = load('model.joblib')

    x = prepare_df(None, countries)[0]

    featImps = {
        'name': x.columns,
        'importance': clf.feature_importances_
    }
    featureImportance = pd.DataFrame(featImps).sort_values(by='importance')

    if printFeats:
        print("Feature Importances:")
        print("Name\t\tImportance")
        for row in featureImportance.iterrows():
            print(row[1]['name'], '\t\t', row[1]['importance'])

    if toFile:
        featureImportance.to_csv(filename, index=False)


def getAccuracy(printAcc=True, countries=True):
    clf = load('model.joblib')

    df = getSafeData(path='../data', countries=countries)
    y = df['Position']
    x = df.drop(['Position', 'URL'], axis=1)
    if not countries:
        x = x.drop(['Region'], axis=1)

    pred = clf.predict(x)
    acc = metrics.accuracy_score(y, pred)

    if printAcc:
        print("Final Accuracy:", acc)

    return acc


def getConfusionMatrix(printCM=True, plot=True, countries=True):
    clf = load('model.joblib')

    df = getSafeData(path='../data', countries=countries)
    y = df['Position']
    x = df.drop(['Position', 'URL'], axis=1)
    if not countries:
        x = x.drop(['Region'], axis=1)

    pred = clf.predict(x)
    cm = metrics.confusion_matrix(y, pred)

    if printCM:
        print("Confusion Matrix:\n", cm)

    if plot:
        plot_confusion_matrix(df, y, pred, normalize=True)

    return cm

getAccuracy()
getConfusionMatrix()