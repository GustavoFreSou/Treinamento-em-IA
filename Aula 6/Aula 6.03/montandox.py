import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
m=len(y)
X = np.column_stack([np.ones(m), x])
beta = np.zeros(X.shape[1])

b0, b1 = 0.0, 0.0
alpha = 0.1

for t in range(1000):
    y_hat = b0 + b1 * x
    erro = y_hat - y
    grad_b0 = erro.sum() / m
    grad_b1 = (erro * x).sum() / m
    b0 = b0 - alpha * grad_b0
    b1 = b1 - alpha * grad_b1

alpha, epocas, tol = 0.1, 1000, 1e-8
historico = []
for t in range(epocas):
    y_hat = X @ beta
    erro = y_hat - y
    custo = (erro @ erro) / (2 * m)
    historico.append(custo)
    gradiente = (X.T @ erro) /m
    beta = beta - alpha * gradiente
    if t > 0 and abs(historico[-2] - historico[-1]) < tol:
        break
print(epocas)

import matplotlib.pyplot as plt

trajetoria = [beta.copy()]
trajetoria.append(beta.copy())
traj = np.array(trajetoria)

B0, B1 = np.meshgrid(np.linspace(-1, 1.5, 200),
                     np.linspace(-0.5, 3, 200))
E = B0[..., None] + B1[..., None] * x - y
J_grade = (E ** 2).sum(axis=-1) / (2 * m)

plt.contour(B0, B1, J_grade, levels=20)
plt.plot(traj[:, 0], traj[:, 1], 'o-')

plt.show()

from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
ax.contour(B0, B1, J_grade, levels=20)
caminho, = ax.plot([], [], 'o-', color='C1')

def atualizar(k):
    caminho.set_data(traj[:k + 1, 0], traj[:k + 1, 1])
    J = ((X @ traj[k] - y) ** 2).sum() / (2 * m)
    ax.set_title(f't = {k}   J = {J:.4f}')
    return caminho, 

anim = FuncAnimation(fig, atualizar, frames=len(traj),
                    interval=100)
anim.save('descida.gif', writer='pillow', fps=10)

fig3d = plt.figure()
ax3d = fig3d.add_subplot(projection='3d')
ax3d.plot_surface(B0, B1, J_grade, cmap='viridis', alpha=0.8)
ax3d.set_xlabel('β₀'); ax3d.set_ylabel('β₁'); ax3d.set_zlabel('J')
plt.show()