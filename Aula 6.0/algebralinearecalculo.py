import numpy as np
import timeit

rng = np.random.default_rng(42)
n,d = 10_000, 5
X = rng.normal(size=(n, d))
w = rng.normal(size=d)
y = X @ w + rng.normal(scale=0.1, size=n)

x = X[0]
total = 0.0
for j in range(d):
    total += x[j] * w[j]

total_v = X[0] @ w
print(np.isclose(total, total_v))

y_hat = np.zeros(n)
for i in range(n):
    for j in range(d):
        y_hat[i] += X[i, j] * w[j]

y_hat_v = X @ w 
np.allclose(y_hat, y_hat_v)
tempo = timeit.timeit(lambda: X @ w, number=1000)
print("Tempo:", tempo)

A,B = X[:200], X[:50]
D = np.zeros((200, 50))
for i in range (200):
    for j in range(50):
        D[i, j] = np.linalg.norm(A[i] - B[j])

diff = A[:, None, :] - B[None, :, :]
D_v = np.sqrt((diff ** 2). sum(axis=2))
np.allclose(D, D_v)
print(np.allclose(D, D_v))

w0 = np.zeros(d)
mse, grad = 0.0, np.zeros(d)
for i in range(n):
    erro = X[i] @ w0 - y[i]
    mse += erro ** 2 / n
    for j in range(d):
        grad[j] += 2 * erro * X[i, j] / n

erro = X @ w0 - y
mse_v = np.mean(erro ** 2)
grad_v = 2 / n * X.T @ erro
np.allclose(grad, grad_v)

def mse_de(w):
    return np.mean((X @ w - y) ** 2)

def grad_numerico(f, w, h=1e-6):
    g = np.zeros_like(w)
    for j in range(len(w)):
        e = np.zeros_like(w)
        e[j] = h
        g[j] = (f(w + e) - f(w - e) / (2 * h))
    return g

np.allclose(grad_v, grad_numerico(mse_de, w0))