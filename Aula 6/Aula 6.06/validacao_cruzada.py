import numpy as np

rng = np.random.default_rng(0)
X0 = rng.normal(loc=[0, 0], scale=1.0, size=(90, 2))
X1 = rng.normal(loc=[2,2], scale=1.0, size=(10, 2))

X = np.vstack([X0, X1])
y = np.concatenate([np.zeros(90, dtype=int), np.ones(10, dtype=int)])

print(X.shape, y.shape)
print(np.unique(y, return_counts=True))

def meu_kfold(n, k=5, shuffle=False, random_state=None):
    indices = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(random_state)
        rng.shuffle(indices)
    tamanhos = np.full(k, n // k, dtype=int)
    tamanhos[: n % k] += 1
    atual = 0
    for t in tamanhos:
        val_idx = indices[atual:atual + t]
        train_idx = np.concatenate([indices[:atual], indices[atual + t:]])
        yield train_idx, val_idx
        atual += t

def meu_stratifield_kfold(y, k=5, shuffle=False, random_state=None):
    y = np.asarray(y)
    rng = np.random.default_rng(random_state)
    partes = [[] for _ in range(k)]
    for c in np.unique(y):
        idx_c = np.where(y == c)[0]
        if shuffle:
            rng.shuffle(idx_c)
        for i, parte in enumerate(np.array_split(idx_c, k)):
            partes[i].append(parte)
    todos = np.arange(len(y))
    for i in range(k):
        val_idx = np.sort(np.concatenate(partes[i]))
        train_idx = np.setdiff1d(todos, val_idx)
        yield train_idx, val_idx

meu_stratified_kfold = meu_stratifield_kfold

def meu_cross_val_score(ModeloClasse, X, y, folds, metrica):
    scores = []
    for train_idx, val_idx in folds:
        modelo = ModeloClasse()
        modelo.fit(X[train_idx], y[train_idx])
        y_pred = modelo.predict(X[val_idx])
        scores.append(metrica(y[val_idx], y_pred))
    return np.array(scores)

CentroideMaisProximo = None  

def contar_classes(gerador):
    for fold, (train_idx, val_idx) in enumerate(gerador, start=1):
        print(fold, np.bincount(y[val_idx], minlength=2))

contar_classes(meu_kfold(100, k=5))
contar_classes(meu_kfold(100, k=5, shuffle=True, random_state=42))
contar_classes(meu_stratifield_kfold(y, k=5, shuffle=True, random_state=42))

class CentroMaisProximo:
    def fit(self, X, y):
        self.class_ = np.unique(y)
        self.centroides_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        dist = np.stack([np.linalg.norm(X - c, axis=1)
                        for c in self.centroides_], axis=1)
        return self.classes_[np.argmin(dist, axist=1)]

def acuracia(y_true, y_pred):
    return np.mean(y_true == y_pred)

folds = meu_stratified_kfold(y, k=5, shuffle=True, random_state=42)
scores = meu_cross_val_score(CentroideMaisProximo, X, y, folds, acuracia)
print(scores, f"{scores.mean():.3f} +- {scores.std():.3f}")

from sklearn.model_selection import(KFold, StratifiedKFold, cross_val_score)
from sklearn.neighbors import NearestCentroid

all(np.array_equal(v1, v2) for (_, v1), (_, v2)
    in zip(meu_kfold(100, k=5), KFold(5).split(X)))

meus_folds = list(meu_stratifield_kfold(y, k=5, shuffle=True, random=state=42))
cross_val_score(NearestCentroid(), X, y, cv=meus_folds)

cv=StratifiedKFold(5, shuffle=True, random_state=42)
cross_val_score(NearestCentroid(), X, y, cv=cv)