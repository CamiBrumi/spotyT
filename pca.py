import pandas as pd
from sklearn.decomposition import PCA



# pca()
# Computes the principal components of our dataset.
# PARAM nr_comp: number of principal components
# RETURN principalDf: the dataframe with the principal components. Each column is a different PC.
#
def pca(nr_comp = 2):
    df = pd.read_csv('data/normalizeddata_train.csv')
    df = df.drop(['Region', 'URL', 'Position'], inplace=False, axis=1)
    pca = PCA(n_components=nr_comp)
    principalComponents = pca.fit_transform(df)

    titles = []
    for i in range(nr_comp):
        titles.append('PC ' + str(i+1))

    principalDf = pd.DataFrame(data=principalComponents
                               , columns=titles)
    return principalDf

# res = pca(2)
# print(res["PC 1"].mean())
# print(res["PC 1"].var())
# 
# print(res["PC 2"].mean())
# print(res["PC 2"].var())