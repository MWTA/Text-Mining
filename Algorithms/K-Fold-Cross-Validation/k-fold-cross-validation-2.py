import pandas as pd
from sklearn.model_selection import KFold

df = pd.read_csv(
    '/home/rodriguesfas/Workspace/Text-Mining/k-fold-cross-validation/kfold.csv')

X = df[['W1', 'W2', 'W3', 'W3']]
y = df['A/N']

kf = KFold(n_splits=10)
for train, test in kf.split(X):
    xtr = X.loc[train]
    ytr = y.loc[train]
    xtest = X.loc[test]
    ytest = y.loc[test]
