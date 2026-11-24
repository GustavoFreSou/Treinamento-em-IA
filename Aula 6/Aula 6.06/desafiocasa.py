import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import NearestCentroid

X, y = load_breast_cancer(return_X_y=True)
print(X.shape, np.unique(y, return_counts=True))

def meu_kfold(n_amostras, k=5, shuffle=False, random_state=None):
    if not 2 <= k <= n_amostras:
        raise ValueError("k deve satisfazer 2 <= k <= n_amostras")
    indices = np.arange(n_amostras)
    if shuffle:
        np.random.default_rng(random_state).shuffle(indices)
    tamanhos = np.full(k, n_amostras // k, dtype=int)
    tamanhos[: n_amostras % k] += 1
    atual = 0
    for t in tamanhos:
        val_idx = indices[atual:atual + t]
        train_idx = np.concatenate([indices[:atual], indices[atual + t:]])
        yield train_idx, val_idx
        atual += t

def meu_stratified_kfold(y, k=5, shuffle=False, random_state=None):
    y = np.asarray(y)
    classes, contagens = np.unique(y, return_counts=True)
    if not 2 <= k <= len(y):
        raise ValueError("k deve satisfazer 2 <= k <= len(y)")
    if k > contagens.min():
        raise ValueError("k maior que o número de amostras da menor classe")
    rng = np.random.default_rng(random_state)
    partes = [[] for _ in range(k)]
    for c in classes:
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

class CentroMaisProximo:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.centroides_ = np.array([X[y == c].mean(axis=0) for c in self.classes_])
        return self

    def predict(self, X):
        dist = np.stack([np.linalg.norm(X - c, axis=1) for c in self.centroides_], axis=1)
        return self.classes_[np.argmin(dist, axis=1)]

def acuracia(y_true, y_pred):
    return np.mean(y_true == y_pred)

def meu_cross_val_score(modelo_factory, X, y, folds, metrica):
    scores = []
    for train_idx, val_idx in folds:
        modelo = modelo_factory()
        modelo.fit(X[train_idx], y[train_idx])
        y_pred = modelo.predict(X[val_idx])
        scores.append(metrica(y[val_idx], y_pred))
    return np.array(scores)

k = 5
folds_kf = list(meu_kfold(len(y), k=k, shuffle=True, random_state=42))
folds_skf = list(meu_stratified_kfold(y, k=k, shuffle=True, random_state=42))

n = len(y)
todas_val = np.concatenate([v for _, v in folds_kf])
assert np.array_equal(np.sort(todas_val), np.arange(n))
assert all(len(np.intersect1d(t, v)) == 0 for t, v in folds_kf + folds_skf)

print("\nÍndices dos folds (K-Fold):")
for i, (t, v) in enumerate(folds_kf, 1):
    print(f"fold {i}: treino={len(t)}, validação={len(v)}, val_idx[:10]={v[:10]}")

print("\nDistribuição de classes na validação (classe 0, classe 1):")
for nome, folds in [("K-Fold", folds_kf), ("Stratified K-Fold", folds_skf)]:
    print(nome)
    for i, (_, v) in enumerate(folds, 1):
        print(f"  fold {i}:", np.bincount(y[v], minlength=2))

scores = meu_cross_val_score(CentroMaisProximo, X, y, folds_skf, acuracia)
ddof = 1
media, desvio = scores.mean(), scores.std(ddof=ddof)
print("\nScores individuais:", np.round(scores, 4))
print(f"Média: {media:.4f} | Desvio-padrão (ddof={ddof}): {desvio:.4f}")

plt.figure(figsize=(7, 4))
plt.bar(range(1, k + 1), scores)
plt.axhline(media, color="red", linestyle="--", label=f"média = {media:.3f}")
plt.xlabel("Fold")
plt.ylabel("Acurácia")
plt.ylim(scores.min() - 0.05, 1.0)
plt.legend()
plt.show()

kf_igual = all(
    np.array_equal(v1, v2)
    for (_, v1), (_, v2) in zip(meu_kfold(n, k=k), KFold(k).split(X))
)
print("\nK-Fold igual ao KFold do scikit-learn (sem shuffle):", kf_igual)

scores_sk = cross_val_score(NearestCentroid(), X, y, cv=folds_skf)
print("Scores meus:       ", np.round(scores, 4))
print("Scores scikit-learn:", np.round(scores_sk, 4))
print("Iguais:", np.allclose(scores, scores_sk))