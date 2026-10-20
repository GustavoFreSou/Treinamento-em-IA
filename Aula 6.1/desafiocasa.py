import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class GradienteDescendente:
    def __init__(self, alpha=0.1, epocas=1000, tol=1e-8):
        self.alpha = alpha
        self.epocas = epocas
        self.tol = tol
        self.beta_ = None
        self.historico_ = []  
        self.betas_ = []       

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        n = X.shape[0]
        Xb = np.c_[np.ones(n), X]            # coluna de 1s (intercepto)
        self.beta_ = np.zeros(Xb.shape[1])   # β começa em zero
        self.historico_ = []
        self.betas_ = [self.beta_.copy()]

        for _ in range(self.epocas):
            erro = Xb @ self.beta_ - y                   # ŷ - y
            custo = (erro @ erro) / (2 * n)              # J(β) = MSE/2
            self.historico_.append(custo)

            grad = (Xb.T @ erro) / n                     # ∇J(β)
            beta_novo = self.beta_ - self.alpha * grad   # passo contra o gradiente
            self.betas_.append(beta_novo.copy())

            # critério de parada: β quase não muda
            if np.linalg.norm(beta_novo - self.beta_) < self.tol:
                self.beta_ = beta_novo
                break
            self.beta_ = beta_novo

        return self

    def predict(self, X):
        # retorna as previsões ŷ
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        Xb = np.c_[np.ones(X.shape[0]), X]
        return Xb @ self.beta_

    def score(self, X, y):
        # retorna o R²
        y = np.asarray(y, dtype=float).ravel()
        y_hat = self.predict(X)
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1 - ss_res / ss_tot

rng = np.random.default_rng(42)
X = rng.uniform(0, 10, (200, 1))
y = 3 + 2 * X.ravel() + rng.normal(0, 0.5, 200)

# ---------- 2) TREINO ----------
modelo = GradienteDescendente(alpha=0.01, epocas=5000)
modelo.fit(X, y)

print("β:", modelo.beta_)                                    # ≈ [3, 2]
print("Custo:", modelo.historico_[0], "->", modelo.historico_[-1])
print("R²:", modelo.score(X, y))                             # ≈ 0.99

# ---------- 3) ANIMAÇÃO ----------
Xb = np.c_[np.ones(len(X)), X]
b0, b1 = np.meshgrid(np.linspace(-1, 6, 100), np.linspace(0, 4, 100))
J = np.array([[np.mean((Xb @ [a, b] - y) ** 2) / 2
               for a, b in zip(r0, r1)] for r0, r1 in zip(b0, b1)])

caminho = np.array(modelo.betas_)[::20]
fig, ax = plt.subplots()
ax.contour(b0, b1, J, levels=30)
linha, = ax.plot([], [], "r.-")

def atualizar(i):
    linha.set_data(caminho[:i, 0], caminho[:i, 1])
    return linha,

anim = FuncAnimation(fig, atualizar, frames=len(caminho), interval=50)
ax.set_xlabel("β₀")
ax.set_ylabel("β₁")
plt.show()