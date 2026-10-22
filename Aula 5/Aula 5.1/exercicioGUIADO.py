from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target

print(X.shape, y.shape)

import pandas as pd

df = pd.DataFrame(X, columns=iris.feature_names)

df['especie'] = iris.target_names[y]

print(df.groupby('especie').mean().round(2))

from sklearn.model_selection import (train_test_split)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.1,
    stratify=y, random_state=42
)

print(X_tr.shape, X_te.shape)

from sklearn.neighbors import (
    KNeighborsClassifier )

modelo = KNeighborsClassifier(
    n_neighbors = 5)
print(modelo.fit(X_tr, y_tr))

y_pred = modelo.predict(X_te)

print(y_pred[:8])
print(y_te[:8])

from sklearn.metrics import(
    accuracy_score, confusion_matrix
)

print(accuracy_score(y_te, y_pred))
print(confusion_matrix(y_te, y_pred))

from sklearn.dummy import DummyClassifier

base = DummyClassifier(
    strategy = 'most_frequent')
base.fit(X_te, y_te)

print(base.score(X_te, y_te))