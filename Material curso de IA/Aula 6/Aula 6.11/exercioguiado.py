import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=4, suppress=True)

iris = load_iris()
X = iris.data.astype(float)
y = iris.target
nomes = iris.feature_names
especies = iris.target_names

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
print("Espécies:", especies)

print("Nomes das características:", nomes)
print("Primeiras 5 linhas:\n", X[:5])
print("Classes e frequências:", np.unique(y, return_counts=True))
assert X.shape == (150, 4)
assert len(np.unique(y)) == 3
pares = [(i, j) for i in range(4) for j in range(i + 1, 4)]
fig, eixos = plt.subplots(2, 3, figsize=(14, 8))
for ax, (i, j) in zip(eixos.flat, pares):
    for classe, especie in enumerate(especies):
        ax.scatter(X[y == classe, i], X[y == classe, j], label=especie, alpha=0.65, s=18)
    ax.set(xlabel=nomes[i], ylabel=nomes[j])
    ax.grid(alpha=0.15)
eixos[0, 0].legend(fontsize=8)
fig.suptitle("Iris — 6 pares das 4 características originais")
fig.tight_layout()
plt.show()

media = X.mean(axis=0)
print("Médias das quatro características:", media)
print("Formato:", media.shape)

X_centralizado = X - media
print("Primeiras 5 linhas centralizadas:\n", X_centralizado[:5])
print("Formato:", X_centralizado.shape)

medias_apos = X_centralizado.mean(axis=0)
print("Médias após centralização:", medias_apos)
assert np.allclose(medias_apos, 0, atol=1e-12)

n = X.shape[0]
cov_manual = X_centralizado.T @ X_centralizado / (n - 1)
print("Matriz de covariância manual:\n", cov_manual)
print("Formato:", cov_manual.shape)
assert cov_manual.shape == (4, 4)

cov_numpy = np.cov(X_centralizado, rowvar=False)
print("Matriz de covariância np.cov:\n", cov_numpy)
print("Coincidem?", np.allclose(cov_manual, cov_numpy))
assert np.allclose(cov_manual, cov_numpy)

fig, ax = plt.subplots(figsize=(6.5, 5))
im = ax.imshow(cov_manual, cmap="viridis")
ax.set_xticks(range(4), [f"x{i+1}" for i in range(4)])
ax.set_yticks(range(4), [f"x{i+1}" for i in range(4)])
ax.set_title("Matriz de covariância do Iris")
fig.colorbar(im, ax=ax, label="Covariância amostral")
plt.show()

autovalores, autovetores = np.linalg.eigh(cov_manual)
print("Autovalores (ordem inicial):", autovalores)
print("Autovetores (colunas):\n", autovetores)
assert np.all(autovalores >= -1e-10)

indices = np.argsort(autovalores)[::-1]
autovalores = autovalores[indices]
autovetores = autovetores[:, indices]
print("Autovalores em ordem decrescente:", autovalores)
print("Primeiro autovetor:", autovetores[:, 0])
assert np.all(np.diff(autovalores) <= 0)

variancia_explicada = autovalores / autovalores.sum()
for i, valor in enumerate(variancia_explicada, start=1):
    print(f"PC{i}: {valor * 100:.2f}%")
print(f"PC1 + PC2: {variancia_explicada[:2].sum() * 100:.2f}%")
assert np.isclose(variancia_explicada.sum(), 1.0)

variancia_acumulada = np.cumsum(variancia_explicada)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, 5), variancia_explicada * 100, alpha=0.8, label="Individual")
ax.plot(range(1, 5), variancia_acumulada * 100, marker="o", color="black", label="Acumulada")
ax.axhline(95, linestyle="--", color="gray", alpha=0.6, label="Referência: 95%")
ax.set(xlabel="Componente principal", ylabel="Variância explicada (%)", xticks=range(1, 5), ylim=(0, 105), title="Variância explicada — Iris sem padronização")
ax.legend()
plt.show()
print("Acumulada (%):", np.round(variancia_acumulada * 100, 2))

k = 2
W = autovetores[:, :k]
print("Matriz W:\n", W)
print("Formato:", W.shape)
print("W.T @ W (deve ser identidade):\n", W.T @ W)
assert W.shape == (4, 2)
assert np.allclose(W.T @ W, np.eye(2))

Z = X_centralizado @ W
print("Formato de Z:", Z.shape)
print("Primeiras 5 flores projetadas:\n", Z[:5])
assert Z.shape == (150, 2)

fig, ax = plt.subplots(figsize=(8, 6))
for classe, especie in enumerate(especies):
    ax.scatter(Z[y == classe, 0], Z[y == classe, 1], label=especie, alpha=0.75, s=32)
ax.set(xlabel=f"PC1 ({variancia_explicada[0]*100:.2f}%)", ylabel=f"PC2 ({variancia_explicada[1]*100:.2f}%)", title="Iris projetado em duas componentes principais")
ax.axhline(0, color="gray", linewidth=0.7, alpha=0.4)
ax.axvline(0, color="gray", linewidth=0.7, alpha=0.4)
ax.legend()
ax.grid(alpha=0.2)
plt.show()

X_reconstruido = Z @ W.T + media
mse_reconstrucao = np.mean((X - X_reconstruido) ** 2)
print("Formato reconstruído:", X_reconstruido.shape)
print("Primeira flor original:", X[0])
print("Primeira flor reconstruída:", X_reconstruido[0])
print(f"MSE de reconstrução: {mse_reconstrucao:.6f}")
assert X_reconstruido.shape == X.shape

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(X[0], "o-", label="Primeira flor — original")
ax.plot(X_reconstruido[0], "s--", label="Primeira flor — reconstruída")
ax.set(xticks=range(4), xticklabels=["Sépala L", "Sépala W", "Pétala L", "Pétala W"], ylabel="Medida (cm)", title="Reconstrução com dois componentes")
ax.legend()
ax.grid(alpha=0.2)
plt.show()

U, S, Vt = np.linalg.svd(X_centralizado, full_matrices=False)
valores_svd = S**2 / (n - 1)
W_svd = Vt[:2].T
Z_svd = X_centralizado @ W_svd
X_reconstruido_svd = Z_svd @ W_svd.T + media
print("Singular values:", S)
print("Autovalores via SVD:", valores_svd)
print("Equivalência de autovalores:", np.allclose(valores_svd, autovalores))
print("Reconstruções equivalentes:", np.allclose(X_reconstruido_svd, X_reconstruido))
assert np.allclose(valores_svd, autovalores, atol=1e-10)
assert np.allclose(X_reconstruido_svd, X_reconstruido, atol=1e-10)

pca_sklearn = PCA(n_components=2)
Z_sklearn = pca_sklearn.fit_transform(X)
X_reconstruido_sklearn = pca_sklearn.inverse_transform(Z_sklearn)

print("Formas — manual, SVD e sklearn:", Z.shape, Z_svd.shape, Z_sklearn.shape)
print("Variância explicada manual:", np.round(variancia_explicada[:2], 6))
print("Variância explicada sklearn:", np.round(pca_sklearn.explained_variance_ratio_, 6))
print("Reconstruções manual vs. sklearn:", np.allclose(X_reconstruido, X_reconstruido_sklearn))
assert np.allclose(variancia_explicada[:2], pca_sklearn.explained_variance_ratio_)
assert np.allclose(X_reconstruido, X_reconstruido_sklearn, atol=1e-10)

fig, eixos = plt.subplots(1, 2, figsize=(12, 5))
for classe, especie in enumerate(especies):
    eixos[0].scatter(Z[y == classe, 0], Z[y == classe, 1], label=especie, s=25, alpha=0.7)
    eixos[1].scatter(Z_sklearn[y == classe, 0], Z_sklearn[y == classe, 1], label=especie, s=25, alpha=0.7)
eixos[0].set(title="PCA manual (NumPy)", xlabel="PC1", ylabel="PC2")
eixos[1].set(title="PCA Scikit-learn", xlabel="PC1", ylabel="PC2")
for eixo in eixos:
    eixo.grid(alpha=0.2)
    eixo.legend()
plt.suptitle("Diferenças de orientação/sinal dos eixos são equivalentes")
plt.tight_layout()
plt.show()

padronizador = StandardScaler()
X_padronizado = padronizador.fit_transform(X)
pca_padronizado = PCA(n_components=4)
pca_padronizado.fit(X_padronizado)

variancia_original = variancia_explicada * 100
variancia_padronizada = pca_padronizado.explained_variance_ratio_ * 100

fig, ax = plt.subplots(figsize=(8, 5))
eixos_x = np.arange(4)
largura = 0.36
ax.bar(eixos_x - largura/2, variancia_original, largura, label="Sem padronização")
ax.bar(eixos_x + largura/2, variancia_padronizada, largura, label="Com StandardScaler")
ax.set(xticks=eixos_x, xticklabels=["PC1", "PC2", "PC3", "PC4"], xlabel="Componente", ylabel="Variância explicada (%)", title="Efeito da padronização no PCA")
ax.legend()
ax.grid(axis="y", alpha=0.2)
plt.show()

print(f"PC1 + PC2 sem padronização: {variancia_explicada[:2].sum()*100:.2f}%")
print(f"PC1 + PC2 com StandardScaler: {pca_padronizado.explained_variance_ratio_[:2].sum()*100:.2f}%")
print("Conclusão: PCA aprende direções de maior variância; os resultados dependem da escala das características.")

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

modelo_supervisionado = make_pipeline(
    PCA(n_components=2),
    LogisticRegression(max_iter=1000)
)
modelo_supervisionado.fit(X_treino, y_treino)
print(f"Acurácia no teste (exemplo com Pipeline): {modelo_supervisionado.score(X_teste, y_teste):.4f}")

