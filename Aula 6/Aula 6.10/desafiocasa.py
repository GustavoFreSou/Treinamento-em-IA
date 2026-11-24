import numpy as np
import matplotlib.pyplot as plt

print("NumPy", np.__version__)

def uniformes(n, semente):

    i = np.arange(1, n + 1)
    v = np.sin(i * 12.9898 + semente * 78.233) * 43758.5453
    return v - np.floor(v)

def ruido_normal(n, semente):

    u = uniformes(3 * n, semente).reshape(3, n)
    return (u.sum(axis=0) - 1.5) * 2

print(uniformes(3, 0).round(3))

centros = np.array([[2.0, 60.0], [9.0, 75.0], [4.5, 260.0]])
desvios = np.array([[0.6, 15.0], [1.0, 18.0], [0.9, 35.0]])
n = 50

blocos = []
for g in range(3):
    s = 10 * g
    freq = centros[g, 0] + desvios[g, 0] * ruido_normal(n, s + 1)
    gasto = centros[g, 1] + desvios[g, 1] * ruido_normal(n, s + 2)
    blocos.append(np.column_stack([freq, gasto]))

X = np.vstack(blocos)

print(X.shape)
print(X[:3].round(1))

assert X.shape == (150, 2), "X deve ter shape (150, 2)"
assert not np.isnan(X).any()

np.savetxt("dataset_clientes.csv", X, delimiter=",",
           header="compras_por_mes,gasto_medio", comments="", fmt="%.4f")

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], s=20, color="gray")
plt.xlabel("Compras por mês")
plt.ylabel("Gasto médio (R$)")
plt.title("Clientes sem rótulos")
plt.show()

media = X.mean(axis=0)
desvio = X.std(axis=0)
desvio = np.where(desvio == 0, 1.0, desvio)
X_esc = (X - media) / desvio

print("média:", media.round(3), "desvio:", desvio.round(3))
assert np.allclose(X_esc.mean(axis=0), 0, atol=1e-9)
assert np.allclose(X_esc.std(axis=0), 1)

K = 3

def inicializar_centroides(X, K, semente):

    indices = np.argsort(uniformes(len(X), semente))[:K]
    return X[indices].copy()

centroides = inicializar_centroides(X_esc, K, semente=42)
print(centroides.round(3))
assert centroides.shape == (K, 2)
assert len({tuple(c) for c in centroides}) == K, "centroides devem ser distintos"
assert not np.shares_memory(centroides, X_esc), "devolva uma cópia"

def calcular_distancias(X, centroides):

    diferencas = X[:, None, :] - centroides[None, :, :]
    return np.sum(diferencas ** 2, axis=2)

D = calcular_distancias(X_esc, centroides)
print(D.shape)
print(D[:3].round(3))
assert D.shape == (150, K)
assert np.isclose(D[0, 0], np.sum((X_esc[0] - centroides[0]) ** 2))

def atribuir_clusters(distancias):

    return np.argmin(distancias, axis=1)

labels = atribuir_clusters(D)
print(labels[:3])
print(np.bincount(labels, minlength=K))
assert labels.shape == (150,)
assert set(labels) <= set(range(K))

def atualizar_centroides(X, labels, centroides):

    novos = centroides.copy()
    for k in range(len(centroides)):
        pontos = X[labels == k]
        if len(pontos) > 0:
            novos[k] = pontos.mean(axis=0)
    return novos

novos = atualizar_centroides(X_esc, labels, centroides)
print(novos.round(3))
k0 = labels == 0
if k0.any():
    assert np.allclose(novos[0], X_esc[k0].mean(axis=0))

def calcular_inercia(distancias, labels):

    return distancias[np.arange(len(labels)), labels].sum()

C = inicializar_centroides(X_esc, K, semente=42)
for it in range(1, 6):
    D = calcular_distancias(X_esc, C)
    labels = atribuir_clusters(D)
    J = calcular_inercia(D, labels)
    print(f"iteração {it}: inércia = {J:.3f}")
    C = atualizar_centroides(X_esc, labels, C)

tol = 1e-4
max_iter = 100
C = inicializar_centroides(X_esc, K, semente=42)

for it in range(1, max_iter + 1):
    D = calcular_distancias(X_esc, C)
    labels = atribuir_clusters(D)
    novos = atualizar_centroides(X_esc, labels, C)
    desloc = np.sqrt(((novos - C) ** 2).sum(axis=1)).max()
    print(f"iteração {it}: deslocamento máximo = {desloc:.4f}")
    C = novos
    if desloc <= tol:
        print("convergiu")
        break

print("iterações usadas:", it)

def atualizar_centroides(X, labels, centroides, distancias):

    novos = centroides.copy()
    d_atual = distancias[np.arange(len(X)), labels].copy()
    for k in range(len(centroides)):
        pontos = X[labels == k]
        if len(pontos) > 0:
            novos[k] = pontos.mean(axis=0)
        else:
            mais_longe = np.argmax(d_atual)
            novos[k] = X[mais_longe]
            d_atual[mais_longe] = -1.0
    return novos

X_teste = np.array([[0, 0], [0, 1], [10, 10], [10, 11]], dtype=float)
C_teste = np.array([[0, 0.5], [10, 10.5], [50, 50]])
D_teste = calcular_distancias(X_teste, C_teste)
L_teste = atribuir_clusters(D_teste)
print(L_teste, np.bincount(L_teste, minlength=3))

C_novo = atualizar_centroides(X_teste, L_teste, C_teste, D_teste)
print(C_novo)
assert not np.isnan(C_novo).any(), "há NaN nos centroides"
assert not np.allclose(C_novo[2], C_teste[2]), "o cluster vazio deveria ter sido realocado"

class KMeansDoZero:
    def __init__(self, n_clusters=3, max_iter=100,
                 tol=1e-4, random_state=0):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        if not 1 <= self.n_clusters <= len(X):
            raise ValueError("n_clusters deve estar entre 1 e o número de amostras")
        if self.max_iter < 1:
            raise ValueError("max_iter deve ser >= 1")

        C = inicializar_centroides(X, self.n_clusters, self.random_state)
        self.historico_centroides_ = [C.copy()]
        self.historico_inercia_ = []

        for it in range(1, self.max_iter + 1):
            D = calcular_distancias(X, C)
            labels = atribuir_clusters(D)
            self.historico_inercia_.append(calcular_inercia(D, labels))
            novos = atualizar_centroides(X, labels, C, D)
            desloc = np.sqrt(((novos - C) ** 2).sum(axis=1)).max()
            C = novos
            self.historico_centroides_.append(C.copy())
            if desloc <= self.tol:
                break

        D = calcular_distancias(X, C)
        self.centroides_ = C
        self.labels_ = atribuir_clusters(D)
        self.inercia_ = calcular_inercia(D, self.labels_)
        self.historico_inercia_.append(self.inercia_)
        self.n_iter_ = it
        return self

    def predict(self, X):
        if not hasattr(self, "centroides_"):
            raise RuntimeError("chame fit() antes de predict()")
        D = calcular_distancias(np.asarray(X, dtype=float), self.centroides_)
        return atribuir_clusters(D)

    def fit_predict(self, X):
        return self.fit(X).labels_

modelo = KMeansDoZero(n_clusters=3, random_state=42)
modelo.fit(X_esc)
print("iterações:", modelo.n_iter_, "| inércia:", round(modelo.inercia_, 3))
print("clientes por cluster:", np.bincount(modelo.labels_))
print("histórico:", np.round(modelo.historico_inercia_, 3))

D_final = calcular_distancias(X_esc, modelo.centroides_)
assert np.array_equal(modelo.labels_, atribuir_clusters(D_final)), "labels_ inconsistente com centroides_"
assert np.isclose(modelo.inercia_, calcular_inercia(D_final, modelo.labels_)), "inercia_ inconsistente"
assert not np.isnan(modelo.centroides_).any()

novos_clientes = np.array([[6, 200], [1.5, 50], [10, 90]])
print("predict:", modelo.predict((novos_clientes - media) / desvio))
assert np.array_equal(KMeansDoZero(3, random_state=42).fit_predict(X_esc), modelo.labels_), "não reprodutível"

m1 = KMeansDoZero(1).fit(X_esc)
assert np.allclose(m1.centroides_, X_esc.mean(axis=0)), "K=1 deve dar a média global"

try:
    KMeansDoZero(5).fit(X_esc[:3])
    raise AssertionError("K > n deveria levantar ValueError")
except ValueError as erro:
    print("K>n:", erro)

X_rep = np.array([[0, 0]] * 3 + [[5, 5]] * 3, dtype=float)
m_rep = KMeansDoZero(3, random_state=0).fit(X_rep)
assert not np.isnan(m_rep.centroides_).any(), "NaN com pontos repetidos"

m_lim = KMeansDoZero(3, max_iter=1, random_state=42).fit(X_esc)
assert m_lim.n_iter_ == 1

H = modelo.historico_centroides_
assert not np.shares_memory(H[0], H[-1]), "o histórico deve guardar cópias"
print("todos os testes da classe passaram")

cores = np.array(["#3B82F6", "#F59E0B", "#10B981"])

C_orig = modelo.centroides_ * desvio + media

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=cores[modelo.labels_], s=20)
plt.scatter(C_orig[:, 0], C_orig[:, 1], marker="X",
            s=250, c=cores, edgecolor="black")
plt.xlabel("Compras por mês")
plt.ylabel("Gasto médio (R$)")
plt.title("Segmentação de clientes (K=3)")
plt.savefig("grafico_clusters.png", dpi=150, bbox_inches="tight")
plt.show()

print("Centroides finais (compras/mês, gasto R$):")
print(C_orig.round(1))

H = modelo.historico_centroides_
fig, eixos = plt.subplots(1, len(H), figsize=(4 * len(H), 3.5))
eixos = np.atleast_1d(eixos)
for t, (ax, C_t) in enumerate(zip(eixos, H)):
    rot = atribuir_clusters(calcular_distancias(X_esc, C_t))
    ax.scatter(X_esc[:, 0], X_esc[:, 1], c=cores[rot], s=10)
    ax.scatter(C_t[:, 0], C_t[:, 1], marker="X", s=200, c=cores, edgecolor="black")
    ax.set_title(f"centroides C{t}")
plt.show()

h = modelo.historico_inercia_
plt.figure(figsize=(6, 4))
plt.plot(range(len(h)), h, marker="o")
plt.xlabel("centroides C_t")
plt.ylabel("inércia")
plt.title("Evolução da inércia")
plt.savefig("grafico_inercia.png", dpi=150, bbox_inches="tight")
plt.show()

assert all(h[i] >= h[i + 1] - 1e-9 for i in range(len(h) - 1)), "a inércia não pode aumentar"

valores_K = range(1, 9)
inerciais = []
for k in valores_K:
    melhor_k = min(KMeansDoZero(k, random_state=s).fit(X_esc).inercia_
                   for s in range(10))
    inerciais.append(melhor_k)

print(np.round(inerciais, 1))

plt.figure(figsize=(6, 4))
plt.plot(valores_K, inerciais, marker="o")
plt.xlabel("K")
plt.ylabel("inércia")
plt.title("Método do cotovelo")
plt.savefig("grafico_cotovelo.png", dpi=150, bbox_inches="tight")
plt.show()

for semente in [42, 1, 7]:
    m = KMeansDoZero(3, random_state=semente).fit(X_esc)
    print(f"semente {semente:2d}: inércia = {m.inercia_:7.3f} | "
          f"iterações = {m.n_iter_} | "
          f"tamanhos = {np.bincount(m.labels_, minlength=3)}")

modelos = [KMeansDoZero(3, random_state=s).fit(X_esc) for s in range(10)]
melhor = min(modelos, key=lambda m: m.inercia_)
print("melhor inércia entre 10 sementes:", round(melhor.inercia_, 3))

from sklearn.cluster import KMeans

sk = KMeans(n_clusters=3, n_init=10, random_state=42).fit(X_esc)
print("inércia sklearn:", round(sk.inertia_, 3))
print("inércia nossa:  ", round(melhor.inercia_, 3))

par = atribuir_clusters(
    calcular_distancias(melhor.centroides_, sk.cluster_centers_))
print("pareamento:", par)

dif = np.abs(melhor.centroides_ - sk.cluster_centers_[par]).max()
print("maior diferença:", round(dif, 6))

concordancia = (par[melhor.labels_] == sk.labels_).mean()
print("concordância:", concordancia)

C_orig = melhor.centroides_ * desvio + media
tamanhos = np.bincount(melhor.labels_, minlength=3)

print(f"{'cluster':>8} | {'clientes':>8} | {'compras/mês':>11} | {'gasto (R$)':>10}")
for k in range(3):
    print(f"{k:>8} | {tamanhos[k]:>8} | {C_orig[k, 0]:>11.1f} | {C_orig[k, 1]:>10.1f}")

checks = {
    "X tem shape (150, 2)": X.shape == (150, 2),
    "sem NaN em X_esc": not np.isnan(X_esc).any(),
    "sem NaN em centroides_": not np.isnan(modelo.centroides_).any(),
    "labels_ consistente com centroides_": np.array_equal(
        modelo.labels_,
        atribuir_clusters(calcular_distancias(X_esc, modelo.centroides_))),
    "inercia_ consistente com centroides_": np.isclose(
        modelo.inercia_,
        np.sum((X_esc - modelo.centroides_[modelo.labels_]) ** 2)),
    "inércia não crescente": all(h[i] >= h[i + 1] - 1e-9 for i in range(len(h) - 1)),
    "cotovelo com 8 pontos": len(inerciais) == 8,
}
for nome, ok in checks.items():
    print("OK " if ok else "FALHOU", nome)
assert all(checks.values()), "Algum critério de aceite falhou"
print("\nDesafio concluído!")