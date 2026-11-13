import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter
from sklearn.model_selection import train_test_split
 
rng = np.random.default_rng(42)
X = np.vstack([rng.normal([2, 2], 1.2, (40, 2)),
               rng.normal([4, 4], 1.2, (40, 2))])
y = np.array([0] * 40 + [1] * 40)
 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
 
def distancia(a, b):
    return np.sum(np.abs(a - b))
 
def distancias_loop(X_train, x):
    return np.array([distancia(xi, x) for xi in X_train])
 
def distancias_loops(X_test, X_train):
    return np.array([distancias_loop(X_train, x) for x in X_test])
 
def distancias_broadcasting(X_test, X_train):
    dif = X_test[:, None, :] - X_train[None, :, :]
    return np.sum(np.abs(dif), axis=2)
 
def votar(rotulos):
    contagem = np.bincount(rotulos)
    return np.flatnonzero(contagem == contagem.max())[0]
 
def predict_loop(X_train, y_train, X_test, k=3):
    previsoes = []
    for x in X_test:
        indices = np.argsort(distancias_loop(X_train, x), kind='stable')[:k]
        previsoes.append(votar(y_train[indices]))
    return np.array(previsoes)
 
def predict_broadcasting(X_train, y_train, X_test, k=3):
    d = distancias_broadcasting(X_test, X_train)
    indices = np.argsort(d, axis=1, kind='stable')[:, :k]
    return np.array([votar(v) for v in y_train[indices]])
 
def acuracia(y_real, y_prev):
    return np.mean(y_real == y_prev)
 
def padronizar(X_train, X_test):
    mu = X_train.mean(axis=0)
    sigma = X_train.std(axis=0)
    return (X_train - mu) / sigma, (X_test - mu) / sigma
 
def medir(funcao, *args, repeticoes=7):
    tempos = []
    for _ in range(repeticoes):
        inicio = perf_counter()
        funcao(*args)
        tempos.append(perf_counter() - inicio)
    return np.median(tempos)
 
for k in [1, 3, 5, 7, 15]:
    p_loop = predict_loop(X_train, y_train, X_test, k)
    p_broad = predict_broadcasting(X_train, y_train, X_test, k)
    print(k, acuracia(y_test, p_loop), acuracia(y_test, p_broad),
          np.array_equal(p_loop, p_broad))
 
k = 5
print(np.allclose(distancias_loops(X_test, X_train),
                  distancias_broadcasting(X_test, X_train)))
 
t_loop = medir(predict_loop, X_train, y_train, X_test, k)
t_broad = medir(predict_broadcasting, X_train, y_train, X_test, k)
print(t_loop, t_broad, t_loop / t_broad)
 
plt.bar(['loops', 'broadcasting'], [t_loop, t_broad], color=['tab:orange', 'tab:green'])
plt.ylabel('tempo mediano (s)')
plt.yscale('log')
plt.show()
 
X_train_s, X_test_s = padronizar(X_train, X_test)
for k in [1, 3, 5, 7, 15]:
    sem = acuracia(y_test, predict_broadcasting(X_train, y_train, X_test, k))
    com = acuracia(y_test, predict_broadcasting(X_train_s, y_train, X_test_s, k))
    print(k, sem, com)
 
def plotar_fronteira(ax, X_tr, y_tr, k, titulo):
    xx, yy = np.meshgrid(
        np.linspace(X_tr[:, 0].min() - 1, X_tr[:, 0].max() + 1, 200),
        np.linspace(X_tr[:, 1].min() - 1, X_tr[:, 1].max() + 1, 200))
    grade = np.c_[xx.ravel(), yy.ravel()]
    z = predict_broadcasting(X_tr, y_tr, grade, k).reshape(xx.shape)
    ax.contourf(xx, yy, z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X_tr[y_tr == 0, 0], X_tr[y_tr == 0, 1], marker='o', label='classe 0')
    ax.scatter(X_tr[y_tr == 1, 0], X_tr[y_tr == 1, 1], marker='^', label='classe 1')
    ax.set_title(titulo)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
 
ks = [1, 5, 15]
fig, eixos = plt.subplots(2, 3, figsize=(15, 9))
for j, k in enumerate(ks):
    plotar_fronteira(eixos[0, j], X_train, y_train, k, f'sem escalonamento, k={k}')
    plotar_fronteira(eixos[1, j], X_train_s, y_train, k, f'com escalonamento, k={k}')
eixos[0, 0].legend()
plt.tight_layout()
plt.show()