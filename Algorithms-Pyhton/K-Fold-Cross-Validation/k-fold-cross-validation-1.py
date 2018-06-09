"""
    A Gentle Introduction to k-fold Cross-Validation
    https://machinelearningmastery.com/k-fold-cross-validation/
    21/05/2018
"""

# scikit-learn k-fold cross-validation
from numpy import array
from sklearn.model_selection import KFold

# data sample
# data = array(
#    [
#        ('bom', 1),
#        ('ruim', 2),
#        ('pessimo', 3),
#        ('triste', 4),
#        ('otimo', 5)
#    ]
# )

data = array(
    [
        'split1.csv',
        'split2.csv',
        'split3.csv',
        'split4.csv',
        'split5.csv',
    ]
)

# prepare cross validation
kfold = KFold(5, True, 1)

# enumerate splits
for train, test in kfold.split(data):
    print('train: %s, test: %s' % (data[train], data[test]))
