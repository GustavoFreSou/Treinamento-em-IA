import numpy as np
rng = np.random.default_rng(42)
X = np.vstack([rng.normal([2, 2], 1.2, (40, 2)),
               rng.normal([4, 4], 1.2, (40, 2))])
y = np.array([0] * 40 + [1] * 40)

import matplotlib.pyplot as plt
plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', label='classe 0')
plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='^', label='classe 1')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

print(X.shape, y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)
print(X_train.shape, X_test.shape)

a, b = X_test[0], X_train[0]
print(np.sqrt(np.sum((a - b) ** 2)))
print(np.sum(np.abs(a - b)))

def distancia(a, b, metrica='euclidiana'):
    if metrica == 'euclidiana':
        return np.sqrt(np.sum((a - b) ** 2))
    return np.sum(np.abs(a - b))

def distancias_loop(X_train, x):
    return np.array([distancia(xi, x) for xi in X_train])

k = 3
distancias = distancias_loop(X_train, X_test[0])
indices = np.argsort(distancias, kind='stable')[:k]
vizinhos = y_train[indices]
print(indices, vizinhos)

votos = np.bincount(vizinhos)
print(votos, votos.argmax())

def predict_loop(X_train, y_train, X_test, k=3):
    previsoes = []
    for x in X_test:
        indices = np.argsort(distancias_loop(X_train, x), kind='stable')[:k]
        previsoes.append(np.bincount(y_train[indices]).argmax())
    return np.array(previsoes)

previsoes_loop = predict_loop(X_train, y_train, X_test, k)
print(previsoes_loop)

def distancias_broadcasting(X_test, X_train):
    dif = X_test[:, None, :] - X_train[None, :, :]
    return np.sqrt(np.sum(dif ** 2, axis=2))

class KNNDoZero:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train, self.y_train = X, y
        return self

    def predict(self, X):
        d = distancias_broadcasting(X, self.X_train)
        indices = np.argsort(d, axis=1, kind='stable')[:, :self.k]
        return np.array([np.bincount(v).argmax() for v in self.y_train[indices]])

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

previsoes_broad = KNNDoZero(k).fit(X_train, y_train).predict(X_test)
print(previsoes_broad)

def distancias_loops(X_test, X_train):
    return np.array([distancias_loop(X_train, x) for x in X_test])

print(np.allclose(distancias_loops(X_test, X_train),
                  distancias_broadcasting(X_test, X_train)))
print(np.array_equal(previsoes_loop, previsoes_broad))

from time import perf_counter
def medir(funcao, *args, repeticoes=7):
    tempos = []
    for _ in range(repeticoes):
        inicio = perf_counter()
        funcao(*args)
        tempos.append(perf_counter() - inicio)
    return np.median(tempos)




