import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

np.set_printoptions(precision=4, suppress=True)

wine = load_wine()
X = wine.data.astype(float)
y = wine.target
nomes = wine.feature_names
classes = wine.target_names

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
print("Classes:", classes)
print("Número de características:", len(nomes))
print("Nomes das características:", nomes)
print("Primeiras 3 linhas:\n", X[:3])
print("Classes e frequências:", np.unique(y, return_counts=True))
assert X.shape == (178, 13)
assert len(np.unique(y)) == 3
assert not np.isnan(X).any()

desvios = X.std(axis=0)
print("Menor desvio:", nomes[int(np.argmin(desvios))], desvios.min().round(4))
print("Maior desvio:", nomes[int(np.argmax(desvios))], desvios.max().round(4))

fig, eixos = plt.subplots(1, 2, figsize=(14, 5))
for classe, nome_classe in enumerate(classes):
    eixos[0].scatter(X[y == classe, 0], X[y == classe, 6], label=nome_classe, alpha=0.7, s=25)
eixos[0].set(xlabel=nomes[0], ylabel=nomes[6], title="Duas características originais")
eixos[0].legend()
eixos[0].grid(alpha=0.2)
eixos[1].bar(range(13), desvios, edgecolor="black")
eixos[1].set(xticks=range(13), xlabel="Característica", ylabel="Desvio padrão (escala log)", yscale="log", title="Escalas muito diferentes")
eixos[1].set_xticklabels(nomes, rotation=75, ha="right")
eixos[1].grid(axis="y", alpha=0.2)
fig.tight_layout()
plt.show()

media = X.mean(axis=0)
print("Médias das características:", media)
print("Formato:", media.shape)

X_centralizado = X - media
print("Primeiras 3 linhas centralizadas:\n", X_centralizado[:3])
print("Formato:", X_centralizado.shape)

medias_apos = X_centralizado.mean(axis=0)
print("Maior média absoluta após centralização:", np.abs(medias_apos).max())
assert np.allclose(medias_apos, 0, atol=1e-10)

n = X.shape[0]
cov_manual = X_centralizado.T @ X_centralizado / (n - 1)
print("Formato da covariância manual:", cov_manual.shape)
print("Traço (variância total):", np.trace(cov_manual).round(4))
assert cov_manual.shape == (13, 13)
assert np.allclose(cov_manual, cov_manual.T)

cov_numpy = np.cov(X_centralizado, rowvar=False)
print("Coincidem com np.cov?", np.allclose(cov_manual, cov_numpy))
assert np.allclose(cov_manual, cov_numpy)

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(cov_manual, cmap="viridis")
ax.set_xticks(range(13), [f"x{i+1}" for i in range(13)])
ax.set_yticks(range(13), [f"x{i+1}" for i in range(13)])
ax.set_title("Matriz de covariância do Wine (sem padronizar)")
fig.colorbar(im, ax=ax, label="Covariância amostral")
plt.show()

autovalores, autovetores = np.linalg.eigh(cov_manual)
print("Autovalores (ordem inicial):", autovalores)
assert np.all(autovalores >= -1e-10)

indices = np.argsort(autovalores)[::-1]
autovalores = autovalores[indices]
autovetores = autovetores[:, indices]
print("Autovalores em ordem decrescente:", autovalores)
print("Primeiro autovetor:", autovetores[:, 0])
assert np.all(np.diff(autovalores) <= 0)
assert np.allclose(autovetores.T @ autovetores, np.eye(13))
assert np.isclose(autovalores.sum(), np.trace(cov_manual))

variancia_explicada = autovalores / autovalores.sum()
for i, valor in enumerate(variancia_explicada[:5], start=1):
    print(f"PC{i}: {valor * 100:.2f}%")
print(f"PC1 + PC2: {variancia_explicada[:2].sum() * 100:.2f}%")
assert np.isclose(variancia_explicada.sum(), 1.0)

variancia_acumulada = np.cumsum(variancia_explicada)
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(1, 14), variancia_explicada * 100, alpha=0.8, label="Individual")
ax.plot(range(1, 14), variancia_acumulada * 100, marker="o", color="black", label="Acumulada")
ax.axhline(95, linestyle="--", color="gray", alpha=0.6, label="Referência: 95%")
ax.set(xlabel="Componente principal", ylabel="Variância explicada (%)", xticks=range(1, 14), ylim=(0, 105), title="Variância explicada — Wine sem padronização")
ax.legend()
plt.show()
print("Acumulada (%):", np.round(variancia_acumulada[:5] * 100, 2))

k = 2
W = autovetores[:, :k]
print("Formato de W:", W.shape)
print("W.T @ W (deve ser identidade):\n", W.T @ W)
assert W.shape == (13, 2)
assert np.allclose(W.T @ W, np.eye(2))

Z = X_centralizado @ W
print("Formato de Z:", Z.shape)
print("Primeiras 3 amostras projetadas:\n", Z[:3])
print("Variância de cada coluna de Z:", Z.var(axis=0, ddof=1))
print("Autovalores correspondentes:   ", autovalores[:2])
assert Z.shape == (178, 2)
assert np.allclose(Z.var(axis=0, ddof=1), autovalores[:2])
assert abs(np.cov(Z, rowvar=False)[0, 1]) < 1e-8

fig, ax = plt.subplots(figsize=(8, 6))
for classe, nome_classe in enumerate(classes):
    ax.scatter(Z[y == classe, 0], Z[y == classe, 1], label=nome_classe, alpha=0.75, s=32)
ax.set(xlabel=f"PC1 ({variancia_explicada[0]*100:.2f}%)", ylabel=f"PC2 ({variancia_explicada[1]*100:.2f}%)", title="Wine projetado em duas componentes principais")
ax.axhline(0, color="gray", linewidth=0.7, alpha=0.4)
ax.axvline(0, color="gray", linewidth=0.7, alpha=0.4)
ax.legend()
ax.grid(alpha=0.2)
plt.show()

pca_sklearn = PCA(n_components=2)
Z_sklearn = pca_sklearn.fit_transform(X)

print("Formas — manual e sklearn:", Z.shape, Z_sklearn.shape)
print("Variância explicada manual:", np.round(variancia_explicada[:2], 6))
print("Variância explicada sklearn:", np.round(pca_sklearn.explained_variance_ratio_, 6))

sinais = np.sign(np.sum(W * pca_sklearn.components_.T, axis=0))
dif_componentes = np.abs(W * sinais - pca_sklearn.components_.T).max()
dif_projecao = np.abs(Z * sinais - Z_sklearn).max()
print("Sinais relativos:", sinais)
print("Maior diferença nas componentes:", dif_componentes)
print("Maior diferença na projeção:", dif_projecao)
assert np.allclose(variancia_explicada[:2], pca_sklearn.explained_variance_ratio_)
assert dif_componentes < 1e-8
assert dif_projecao < 1e-8

X_reconstruido = Z @ W.T + media
mse_reconstrucao = np.mean((X - X_reconstruido) ** 2)
X_reconstruido_sklearn = pca_sklearn.inverse_transform(Z_sklearn)
print(f"MSE de reconstrução: {mse_reconstrucao:.6f}")
assert X_reconstruido.shape == X.shape
assert np.allclose(X_reconstruido, X_reconstruido_sklearn, atol=1e-8)

fig, eixos = plt.subplots(1, 2, figsize=(13, 5))
for classe, nome_classe in enumerate(classes):
    eixos[0].scatter(Z[y == classe, 0], Z[y == classe, 1], label=nome_classe, s=25, alpha=0.7)
    eixos[1].scatter(Z_sklearn[y == classe, 0], Z_sklearn[y == classe, 1], label=nome_classe, s=25, alpha=0.7)
eixos[0].set(title="PCA manual (NumPy)", xlabel="PC1", ylabel="PC2")
eixos[1].set(title="PCA Scikit-learn", xlabel="PC1", ylabel="PC2")
for eixo in eixos:
    eixo.grid(alpha=0.2)
    eixo.legend()
plt.suptitle("Diferenças de sinal dos eixos são equivalentes")
plt.tight_layout()
plt.show()

padronizador = StandardScaler()
X_padronizado = padronizador.fit_transform(X)
assert np.allclose(X_padronizado.mean(axis=0), 0, atol=1e-10)
assert np.allclose(X_padronizado.std(axis=0), 1)

cov_pad = X_padronizado.T @ X_padronizado / (n - 1)
autovalores_pad, autovetores_pad = np.linalg.eigh(cov_pad)
indices_pad = np.argsort(autovalores_pad)[::-1]
autovalores_pad = autovalores_pad[indices_pad]
autovetores_pad = autovetores_pad[:, indices_pad]
variancia_pad = autovalores_pad / autovalores_pad.sum()
W_pad = autovetores_pad[:, :2]
Z_pad = X_padronizado @ W_pad

pca_pad_sklearn = PCA(n_components=13).fit(X_padronizado)
print("EVR padronizado manual:   ", np.round(variancia_pad[:4], 6))
print("EVR padronizado sklearn:  ", np.round(pca_pad_sklearn.explained_variance_ratio_[:4], 6))
assert np.allclose(variancia_pad, pca_pad_sklearn.explained_variance_ratio_)

fig, ax = plt.subplots(figsize=(10, 5))
eixos_x = np.arange(13)
largura = 0.38
ax.bar(eixos_x - largura/2, variancia_explicada * 100, largura, label="Sem padronização")
ax.bar(eixos_x + largura/2, variancia_pad * 100, largura, label="Com StandardScaler")
ax.set(xticks=eixos_x, xticklabels=[f"PC{i+1}" for i in range(13)], xlabel="Componente", ylabel="Variância explicada (%)", title="Efeito da padronização no PCA do Wine")
ax.legend()
ax.grid(axis="y", alpha=0.2)
plt.show()

fig, eixos = plt.subplots(1, 2, figsize=(14, 5.5))
for ax, Zp, ev, titulo in [(eixos[0], Z, variancia_explicada, "Sem padronização"), (eixos[1], Z_pad, variancia_pad, "Com padronização")]:
    for classe, nome_classe in enumerate(classes):
        ax.scatter(Zp[y == classe, 0], Zp[y == classe, 1], label=nome_classe, alpha=0.75, s=32)
        centro = Zp[y == classe].mean(axis=0)
        ax.scatter(*centro, marker="X", s=220, edgecolors="black", linewidths=1.4, color=f"C{classe}")
        ax.annotate(nome_classe, centro, xytext=(8, 8), textcoords="offset points", fontweight="bold")
    ax.set(xlabel=f"PC1 ({ev[0]*100:.2f}%)", ylabel=f"PC2 ({ev[1]*100:.2f}%)", title=titulo)
    ax.grid(alpha=0.2)
eixos[0].legend()
plt.tight_layout()
plt.show()

print(f"PC1 + PC2 sem padronização: {variancia_explicada[:2].sum()*100:.2f}%")
print(f"PC1 + PC2 com StandardScaler: {variancia_pad[:2].sum()*100:.2f}%")

ordem_pc1 = np.argsort(np.abs(W[:, 0]))[::-1][:3]
ordem_pc1_pad = np.argsort(np.abs(W_pad[:, 0]))[::-1][:3]
ordem_pc2_pad = np.argsort(np.abs(W_pad[:, 1]))[::-1][:3]
print("\nMaiores pesos na PC1 sem padronização:")
for i in ordem_pc1:
    print(f"  {nomes[i]:32s} {W[i, 0]:8.4f}")
print("Maiores pesos na PC1 com padronização:")
for i in ordem_pc1_pad:
    print(f"  {nomes[i]:32s} {W_pad[i, 0]:8.4f}")
print("Maiores pesos na PC2 com padronização:")
for i in ordem_pc2_pad:
    print(f"  {nomes[i]:32s} {W_pad[i, 1]:8.4f}")

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.bar(eixos_x - largura/2, W[:, 0], largura, label="Sem padronização")
ax.bar(eixos_x + largura/2, W_pad[:, 0], largura, label="Com StandardScaler")
ax.set(xticks=eixos_x, ylabel="Peso na PC1", title="Quais características formam a PC1")
ax.set_xticklabels(nomes, rotation=75, ha="right")
ax.legend()
ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()

from sklearn.metrics import silhouette_score

sil_bruto = silhouette_score(Z, y)
sil_pad = silhouette_score(Z_pad, y)
print(f"Silhouette das classes sem padronização: {sil_bruto:.4f}")
print(f"Silhouette das classes com padronização: {sil_pad:.4f}")

dominante = nomes[int(np.argmax(np.abs(W[:, 0])))]
print(f"\nSem padronizar, a PC1 é dominada por '{dominante}', a característica de maior escala.")
print("Com padronização, a variância se distribui e vários componentes químicos contribuem.")
print("Conclusão: PCA aprende direções de maior variância; os resultados dependem da escala das características.")
print("A PCA não usou y, e mesmo assim as três classes de vinho ficam visíveis em 2D quando padronizada.")

assert variancia_pad[0] < variancia_explicada[0]
assert sil_pad > sil_bruto
assert not np.isnan(Z_pad).any()

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

for rotulo, etapas in [
    ("sem padronização", [PCA(n_components=2), LogisticRegression(max_iter=1000)]),
    ("com padronização", [StandardScaler(), PCA(n_components=2), LogisticRegression(max_iter=1000)]),
]:
    modelo = make_pipeline(*etapas)
    modelo.fit(X_treino, y_treino)
    print(f"Acurácia no teste ({rotulo}): {modelo.score(X_teste, y_teste):.4f}")
