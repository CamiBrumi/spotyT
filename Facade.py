"""
This is a facade file
"""
from knn import knn_score

"""
KNN
Best results with:
k = 9 for top 10
folds = 4 (not cross validated)
currently is taking in normalizeddata_train
"""
knn_score(9,10)
# knn_score(9,50)
# knn_score(9,100)
# knn_score(9,150)


"""
Random Forest
Best results with:
No Trees: 84
No Folds: 6
Max Depth: 
"""


"""
SVM
Best results with:
~add info here
"""