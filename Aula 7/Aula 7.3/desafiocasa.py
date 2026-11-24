import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance

dados = load_breast_cancer()
X = pd.DataFrame(dados.data, columns=dados.feature_names)
y = dados.target
nomes = list(X.columns)

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
print("Classes:", dict(zip(dados.target_names, np.bincount(y))))
print("Valores ausentes:", X.isna().sum().sum())
print(X.describe().T[["mean", "std", "min", "max"]].round(3).to_string())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

print(X_train.shape, X_test.shape)
print(np.bincount(y_train), np.bincount(y_test))

arvore = DecisionTreeClassifier(random_state=42)
arvore.fit(X_train, y_train)
pred_arvore = arvore.predict(X_test)
print("Profundidade:", arvore.get_depth())
print("Folhas:", arvore.get_n_leaves())
print("Acc. treino:", accuracy_score(y_train, arvore.predict(X_train)))
print("Acc. teste:", accuracy_score(y_test, pred_arvore))

floresta = RandomForestClassifier(
    n_estimators=500,
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)
t0 = time.perf_counter()
floresta.fit(X_train, y_train)
tempo_treino = time.perf_counter() - t0
t0 = time.perf_counter()
pred_floresta = floresta.predict(X_test)
tempo_previsao = time.perf_counter() - t0
print("Acc. treino:", accuracy_score(y_train, floresta.predict(X_train)))
print("Acc. teste:", accuracy_score(y_test, pred_floresta))
print("Tempo de treino (s):", tempo_treino)
print("Tempo de previsão (s):", tempo_previsao)

quantidades = [1, 5, 10, 25, 50, 100, 200, 500]
resultados_n = []
for n in quantidades:
    usa_oob = n >= 25
    rf_n = RandomForestClassifier(n_estimators=n,
        oob_score=usa_oob, random_state=42, n_jobs=-1)
    rf_n.fit(X_train, y_train)
    resultados_n.append({
        "n_estimators": n,
        "acc_treino": accuracy_score(y_train, rf_n.predict(X_train)),
        "acc_teste": accuracy_score(y_test, rf_n.predict(X_test)),
        "oob": rf_n.oob_score_ if usa_oob else np.nan,
    })
tabela_n = pd.DataFrame(resultados_n)
print(tabela_n.to_string(index=False))

profundidades = [1, 2, 3, 4, 6, 8, 10, None]
resultados_d = []
for d in profundidades:
    a = DecisionTreeClassifier(max_depth=d,
            random_state=42).fit(X_train, y_train)
    f = RandomForestClassifier(n_estimators=500,
            max_depth=d, random_state=42,
            n_jobs=-1).fit(X_train, y_train)
    for nome, m in [("Árvore", a), ("Floresta", f)]:
        resultados_d.append({"modelo": nome,
            "max_depth": str(d),
            "acc_treino": m.score(X_train, y_train),
            "acc_teste": m.score(X_test, y_test)})
tabela_d = pd.DataFrame(resultados_d)
print(tabela_d.to_string(index=False))

print("OOB Score:", floresta.oob_score_)
print("Acc. teste:", accuracy_score(y_test, pred_floresta))
print("Diferença:", floresta.oob_score_ - accuracy_score(y_test, pred_floresta))

imp_mdi = floresta.feature_importances_
perm = permutation_importance(
    floresta,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42
)
tabela_imp = pd.DataFrame({
    "caracteristica": nomes,
    "mdi": imp_mdi,
    "permutacao": perm.importances_mean,
    "permutacao_std": perm.importances_std
}).sort_values("mdi", ascending=False)
print(tabela_imp.head(10).to_string(index=False))

arvore_rasa = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X_train, y_train)
floresta_rasa = RandomForestClassifier(
    n_estimators=500, max_depth=4, bootstrap=True,
    oob_score=True, random_state=42, n_jobs=-1
).fit(X_train, y_train)

modelos = [
    ("Árvore de Decisão",   arvore,        1,   "None", None),
    ("Árvore rasa",         arvore_rasa,   1,   "4",    None),
    ("Random Forest",       floresta,      500, "None", floresta.oob_score_),
    ("Random Forest rasa",  floresta_rasa, 500, "4",    floresta_rasa.oob_score_),
]

linhas = []
for nome, m, n_arv, prof, oob in modelos:
    pred = m.predict(X_test)
    linhas.append({
        "modelo": nome,
        "arvores": n_arv,
        "max_depth": prof,
        "treino": accuracy_score(y_train, m.predict(X_train)),
        "teste": accuracy_score(y_test, pred),
        "matriz_teste": confusion_matrix(y_test, pred).tolist(),
        "oob": oob if oob is not None else "n/a",
    })
tabela_final = pd.DataFrame(linhas)
print(tabela_final.to_string())

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
for ax, (nome, m, _, _, _) in zip(axes, modelos):
    ConfusionMatrixDisplay.from_predictions(
        y_test, m.predict(X_test),
        display_labels=dados.target_names,
        cmap="Purples", colorbar=False, ax=ax
    )
    ax.set_title(nome)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(tabela_n["n_estimators"], tabela_n["acc_treino"], marker="o", label="Treino")
ax.plot(tabela_n["n_estimators"], tabela_n["acc_teste"], marker="o", label="Teste")
ax.plot(tabela_n["n_estimators"], tabela_n["oob"], marker="o", linestyle="--", label="OOB")
ax.set_xscale("log")
ax.set_xlabel("n_estimators (escala log)")
ax.set_ylabel("Acurácia")
ax.legend()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, nome in zip(axes, ["Árvore", "Floresta"]):
    sub = tabela_d[tabela_d["modelo"] == nome]
    ax.plot(sub["max_depth"], sub["acc_treino"], marker="o", label="Treino")
    ax.plot(sub["max_depth"], sub["acc_teste"], marker="o", label="Teste")
    ax.set_title(nome)
    ax.set_xlabel("max_depth")
    ax.set_ylabel("Acurácia")
    ax.legend()
plt.tight_layout()
plt.show()

top = tabela_imp.head(10).iloc[::-1]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].barh(top["caracteristica"], top["mdi"], color="#7255E8")
axes[0].set_title("Impureza (MDI)")
top_perm = tabela_imp.sort_values("permutacao", ascending=False).head(10).iloc[::-1]
axes[1].barh(top_perm["caracteristica"], top_perm["permutacao"],
             xerr=top_perm["permutacao_std"], color="#352364")
axes[1].set_title("Permutação (teste)")
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(7, 4.5))
rotulos = [m[0] for m in modelos]
x = np.arange(len(rotulos))
ax.bar(x - 0.2, tabela_final["treino"], width=0.4, label="Treino", color="#C9BEF5")
ax.bar(x + 0.2, tabela_final["teste"], width=0.4, label="Teste", color="#352364")
ax.set_xticks(x)
ax.set_xticklabels(rotulos, rotation=15)
ax.set_ylim(0.8, 1.02)
ax.set_ylabel("Acurácia")
ax.legend()
plt.tight_layout()
plt.show()
