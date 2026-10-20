import numpy as np


def norma(x, ord=2):
    x = np.asarray(x, dtype=float)
    if ord == 1:
        return np.abs(x).sum()
    return np.sqrt((x ** 2).sum())


def distancias(A, B):
    a2 = (A ** 2).sum(axis=1)[:, None]
    b2 = (B ** 2).sum(axis=1)[None, :]
    d2 = a2 + b2 - 2 * A @ B.T
    return np.sqrt(np.maximum(d2, 0))


def mse(X, y, w):
    erro = X @ w - y
    return np.mean(erro ** 2)


def grad_mse(X, y, w):
    erro = X @ w - y
    return (2 / len(y)) * (X.T @ erro)


def grad_numerico(f, w, h=1e-6):
    g = np.zeros_like(w, dtype=float)
    for j in range(len(w)):
        e = np.zeros_like(w, dtype=float)
        e[j] = h
        g[j] = (f(w + e) - f(w - e)) / (2 * h)
    return g