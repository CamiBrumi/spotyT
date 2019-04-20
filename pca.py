import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

## Result structure
# Stores the output of the PCA.
# variance_ratio: The amount of variance explained by each of the selected components.
# transformed_df: The transformed dataset after the reduction
# pc_summary: summary of the Principal axes in feature space, representing the directions of maximum variance in the
# data. The components are sorted by explained_variance_.
class result:
    def __init__(self):
        self.variance_ratio = []
        self.transformed_df = pd.DataFrame()
        self.pc_summary = pd.DataFrame()

# pca()
# Returns a struct with the results from the PCA, as explained above.
# PARAM nr_comp: number of principal components
# RETURN result: the struct with the results
#
def pca(nr_comp = 2):
    df = pd.read_csv('data/normalizeddata_train.csv')
    df = df.drop(['Region', 'URL', 'Position'], inplace=False, axis=1)
    pca = PCA(n_components=nr_comp)
    new_df = pca.fit_transform(df)

    summary = pd.DataFrame(pca.components_, columns=df.columns, index=['PC-1', 'PC-2'])
    result.variance_ratio = pca.explained_variance_
    result.transformed_df = new_df
    result.pc_summary = summary

    return result


result_pca = pca(2)
print(result_pca.variance_ratio)
print(result_pca.transformed_df)
print(result_pca.pc_summary)