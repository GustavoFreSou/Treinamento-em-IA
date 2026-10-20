import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML  


class GradienteDescendente:
    def __init__(self, alpha=0.1, epocas=1000, tol=1e-8):
        self.alpha = alpha
        self.epocas = epocas
        self.tol = tol
        self.beta_ = None
        self.historico_ = []
        self.trajetoria_ = None

    def _adicionar_intercepto(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.column_stack([np.ones(X.shape[0]), X])

    def fit(self, X, y):
        X = self._adicionar_intercepto(X)
        y = np.asarray(y, dtype=float).ravel()
        m = len(y)
        beta = np.zeros(X.shape[1])
        historico = []
        trajetoria = [beta.copy()]                 # antes do laço
        for t in range(self.epocas):
            y_hat = X @ beta
            erro = y_hat - y
            custo = (erro @ erro) / (2 * m)
            historico.append(custo)                # custo antes de atualizar
            gradiente = (X.T @ erro) / m
            beta = beta - self.alpha * gradiente
            trajetoria.append(beta.copy())         # após atualizar
            if t > 0 and abs(historico[-2] - historico[-1]) < self.tol:
                break
        self.beta_ = beta
        self.historico_ = historico
        self.trajetoria_ = np.array(trajetoria)
        return self

    def predict(self, X):
        return self._adicionar_intercepto(X) @ self.beta_

    def score(self, X, y):
        y = np.asarray(y, dtype=float).ravel()
        y_hat = self.predict(X)
        return 1.0 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)


# ===== Teste =====
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
m = len(x)

g = GradienteDescendente().fit(x, y)
print("beta_:", g.beta_)                                  # ≈ [0, 2]
print("custo diminui:", all(np.diff(g.historico_) <= 0))  # True
print("predict:", g.predict(x))                           # ≈ [2 4 6]
print("R²:", g.score(x, y))                               # ≈ 1.0

# ===== Mapa de J (grade) =====
traj = g.trajetoria_
X = g._adicionar_intercepto(x)

B0, B1 = np.meshgrid(np.linspace(-1, 1.5, 200),
                     np.linspace(-0.5, 3, 200))
E = B0[..., None] + B1[..., None] * x - y
J_grade = (E ** 2).sum(axis=-1) / (2 * m)

# ===== Animação =====
fig, ax = plt.subplots()
ax.contour(B0, B1, J_grade, levels=20)
caminho, = ax.plot([], [], 'o-', color='C1')

def atualizar(k):
    caminho.set_data(traj[:k + 1, 0], traj[:k + 1, 1])
    J = ((X @ traj[k] - y) ** 2).sum() / (2 * m)
    ax.set_title(f't = {k}   J = {J:.4f}')
    return caminho,

anim = FuncAnimation(fig, atualizar, frames=len(traj), interval=100)
anim.save('descida.gif', writer='pillow', fps=10)
HTML(anim.to_jshtml())   # no notebook: mostra a animação na célula