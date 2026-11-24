import numpy as np
import matplotlib.pyplot as plt

class RegressaoLogistica:
    def __init__(self, alpha=0.1, epocas=1000):
        self.alpha = alpha
        self.epocas = epocas
        self.beta = None
        self.historico = []

    def _sigmoid(self, z):
        z = np.clip(z, -30, 30)
        return 1 / (1 + np.exp(-z))

    def _com_intercepto(self, X):
        return np.hstack([np.ones((X.shape[0], 1)), X])

    def _custo(self, p, y, eps=1e-12):
        p = np.clip(p, eps, 1 - eps)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    def fit(self, X, y):
        Xb = self._com_intercepto(X)
        self.beta = np.zeros(Xb.shape[1])
        self.historico = []
        for _ in range(self.epocas):
            p = self._sigmoid(Xb @ self.beta)
            self.historico.append(self._custo(p, y))
            gradiente = Xb.T @ (p - y) / len(y)
            self.beta = self.beta - self.alpha * gradiente
        return self

    def predict_proba(self, X):
        return self._sigmoid(self._com_intercepto(X) @ self.beta)

    def predict(self, X, limiar=0.5):
        return (self.predict_proba(X) >= limiar).astype(int)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

rng = np.random.default_rng(42)
m_por_classe = 50
X0 = rng.normal(loc=[-1, -1], scale=1.2, size=(m_por_classe, 2))
X1 = rng.normal(loc=[1, 1], scale=1.2, size=(m_por_classe, 2))

X = np.vstack([X0, X1])
y = np.concatenate([np.zeros(m_por_classe), np.ones(m_por_classe)])

idx = rng.permutation(len(y))
corte = int(0.8 * len(y))
treino, teste = idx[:corte], idx[corte:]
X_treino, y_treino = X[treino], y[treino]
X_teste, y_teste = X[teste], y[teste]

plt.scatter(X[y == 0, 0], X[y == 0, 1], label="classe 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="classe 1")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()

modelo = RegressaoLogistica(alpha=0.1, epocas=1000).fit(X_treino, y_treino)

print(modelo.predict(X_teste)[:5], y_teste[:5].astype(int))
print(f"acurácia treino {modelo.score(X_treino, y_treino):.3f} | teste {modelo.score(X_teste, y_teste):.3f}")

plt.plot(modelo.historico)
plt.xlabel("época")
plt.ylabel("J(β)")
plt.show()

b0, b1, b2 = modelo.beta
x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
x2 = -(b0 + b1 * x1) / b2

plt.scatter(X[y == 0, 0], X[y == 0, 1], label="classe 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], label="classe 1")
plt.plot(x1, x2, "k--", label="fronteira p = 0.5")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.show()
