import numpy as np

def norma(x, ord=2):
    x = np.asarray(x, dtype=float)
    if ord == 1:
        return np.abs(x).sum()
    if ord == 2:
        return np.sqrt((x ** 2).sum())
    raise ValueError("ord deve ser 1 ou 2")

def distancias(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    na = (A ** 2).sum(axis=1)[:, None]
    nb = (B ** 2).sum(axis=1)[None, :]
    sq = na + nb - 2 * A @ B.T
    D = np.sqrt(np.maximum(sq, 0.0))
    i, j = np.nonzero(sq < 1e-8 * (na + nb))
    D[i, j] = np.sqrt(((A[i] - B[j]) ** 2).sum(axis=1))
    return D

def mse(X, y, w):
    erro = X @ w - y
    return np.mean(erro ** 2)

def grad_mse(X, y, w):
    erro = X @ w - y
    return 2 / len(y) * X.T @ erro

def grad_numerico(f, w, h=1e-6):
    w = np.asarray(w, dtype=float)
    g = np.zeros_like(w)
    for j in range(len(w)):
        e = np.zeros_like(w)
        e[j] = h
        g[j] = (f(w + e) - f(w - e)) / (2 * h)
    return g