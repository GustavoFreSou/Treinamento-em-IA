import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier

X = np.array([
    [19, 1], [25, 6], [22, 6], [28, 8], [24, 2], [31, 3],
    [27, 3], [34, 9], [35, 2], [36, 5], [41, 3], [39, 8],
    [47, 1], [42, 6], [55, 4], [45, 10], [58, 8], [29, 4],
    [61, 6], [48, 7], [64, 9], [33, 11], [38, 4], [52, 9]])
y = np.array([0, 1] * 12)
nomes = ["Idade", "Visitas"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y)

def gini(y):
    if len(y) == 0:
        return 0.0
    _, contagens = np.unique(y, return_counts=True)
    return 1.0 - np.sum((contagens / len(y)) ** 2)

def gini_ponderado(y_esq, y_dir):
    n = len(y_esq) + len(y_dir)
    if n == 0:
        return 0.0
    return len(y_esq) / n * gini(y_esq) + len(y_dir) / n * gini(y_dir)

def melhor_divisao(X, y):
    melhor = (None, None, np.inf)
    for coluna in range(X.shape[1]):
        valores = np.unique(X[:, coluna])
        for limiar in (valores[:-1] + valores[1:]) / 2:
            mascara = X[:, coluna] <= limiar
            imp = gini_ponderado(y[mascara], y[~mascara])
            if imp < melhor[2]:
                melhor = (coluna, limiar, imp)
    return melhor

class No:
    def __init__(self, feature=None, threshold=None, left=None,
                 right=None, classe=None, samples=0, impureza=0.0):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.classe = classe
        self.samples = samples
        self.impureza = impureza

    def eh_folha(self):
        return self.left is None and self.right is None

class ArvoreDecisaoDoZero:
    def __init__(self, max_depth=3, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.raiz = None

    def _criar_folha(self, y):
        classes, contagens = np.unique(y, return_counts=True)
        return No(classe=classes[np.argmax(contagens)],
                  samples=len(y), impureza=gini(y))

    def fit(self, X, y):
        self.raiz = self._construir(np.asarray(X, dtype=float), np.asarray(y), 0)
        return self

    def _construir(self, X, y, depth):
        no = self._criar_folha(y)
        if (no.impureza == 0 or depth >= self.max_depth
                or len(y) < self.min_samples_split):
            return no
        col, lim, imp = melhor_divisao(X, y)
        if col is None or no.impureza - imp <= 1e-12:
            return no
        mascara = X[:, col] <= lim
        no.feature, no.threshold = col, lim
        no.left = self._construir(X[mascara], y[mascara], depth + 1)
        no.right = self._construir(X[~mascara], y[~mascara], depth + 1)
        return no

    def _prever_amostra(self, amostra, no):
        if no.eh_folha():
            return no.classe
        if amostra[no.feature] <= no.threshold:
            return self._prever_amostra(amostra, no.left)
        return self._prever_amostra(amostra, no.right)

    def predict(self, X):
        return np.array([self._prever_amostra(a, self.raiz)
                         for a in np.asarray(X, dtype=float)])

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))

def imprimir_arvore(no, nomes, nivel=0, rotulo="[Raiz]"):
    recuo = "    " * nivel
    if no.eh_folha():
        print(f"{recuo}{rotulo}\n{recuo}    Classe: {no.classe}\n"
              f"{recuo}    Amostras: {no.samples}")
        return
    print(f"{recuo}{rotulo} {nomes[no.feature]} <= {no.threshold:g}?")
    imprimir_arvore(no.left, nomes, nivel + 1, "[Sim]")
    imprimir_arvore(no.right, nomes, nivel + 1, "[Não]")

modelo = ArvoreDecisaoDoZero(max_depth=3).fit(X_train, y_train)
y_pred = modelo.predict(X_test)

print("Real:    ", y_test)
print("Previsto:", y_pred)
print("Acurácia teste:", round(modelo.score(X_test, y_test), 3))
print("Acurácia treino:", round(modelo.score(X_train, y_train), 3))
print(confusion_matrix(y_test, y_pred))
imprimir_arvore(modelo.raiz, nomes)

xx, yy = np.meshgrid(np.linspace(15, 68, 200), np.linspace(0, 12, 200))
Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
             colors=["#F4BA64", "#35D49A"], alpha=0.25)
for c, cor in [(0, "#F4BA64"), (1, "#35D49A")]:
    plt.scatter(X_train[y_train == c, 0], X_train[y_train == c, 1],
                c=cor, label=str(c))
plt.xlabel("Idade (anos)")
plt.ylabel("Visitas à loja no mês")
plt.title("Regiões de decisão · max_depth=3")
plt.legend()
plt.show()

sk = DecisionTreeClassifier(criterion="gini", max_depth=3,
                            min_samples_split=2, random_state=0).fit(X_train, y_train)
y_pred_sk = sk.predict(X_test)
print("Acurácia sklearn:", round(accuracy_score(y_test, y_pred_sk), 3))
print("Mesmas predições:", np.array_equal(y_pred, y_pred_sk))

#imposto