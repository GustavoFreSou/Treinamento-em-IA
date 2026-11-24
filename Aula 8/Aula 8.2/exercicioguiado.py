import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from importlib.metadata import version

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix,
    roc_auc_score
)
from lazypredict.Supervised import LazyClassifier

print("LazyPredict", version("lazypredict"))

dados = load_breast_cancer()
X = dados.data
y = dados.target

print("Formato de X:", X.shape)
print("Formato de y:", y.shape)
print("Classes: ", np.unique(y, return_counts=True))
print("Nomes das classes: ", dados.target_names)
print("Quantidade de características:", len(dados.feature_names))

df = pd.DataFrame(X, columns=dados.feature_names)
df["target"] = y
print(df.head())
print(df.describe().T)

nomes, contagens = np.unique(y, return_counts=True)
rotulos = [dados.target_names[i] for i in nomes]

fig, ax = plt.subplots(figsize=(6, 4))
barras = ax.bar(rotulos, contagens,
                color=["#E17B4D", "#4158A6"])
ax.bar_label(barras)
ax.set_title("Amostras por classe")
ax.set_ylabel("Quantidade")
plt.show()

X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    stratify=y_temp,
    random_state=42
)

print("Treinamento:", X_train.shape, y_train.shape)
print("Validação:  ", X_valid.shape, y_valid.shape)
print("Teste:      ", X_test.shape, y_test.shape)

def mostrar_proporcao(nome, y_conjunto):
    proporcoes = pd.Series(y_conjunto).value_counts(normalize=True).sort_index()
    print(f"\n{nome}")
    for classe, proporcao in proporcoes.items():
        print(f"{dados.target_names[classe]}: {proporcao:.1%}")

mostrar_proporcao("Treinamento", y_train)
mostrar_proporcao("Validação", y_valid)
mostrar_proporcao("Teste", y_test)

baseline = DummyClassifier(strategy="prior")
baseline.fit(X_train, y_train)
pred_baseline = baseline.predict(X_valid)
acc_dummy = accuracy_score(y_valid, pred_baseline)
bal_dummy = balanced_accuracy_score(
    y_valid, pred_baseline
)
print(f"Dummy * Accuracy:        {acc_dummy:.4f}")
print(f"Dummy * Balanced Accuracy:        {bal_dummy:.4f}")

cm_dummy = confusion_matrix(y_valid, pred_baseline, labels=[0, 1])
print(pd.DataFrame(
    cm_dummy,
    index=["Real: malignant", "Real: benign"],
    columns=["Previsto: malignant", "Previsto: benign"]
))

clf = LazyClassifier(
    verbose=0,
    ignore_warnings=True,
    custom_metric=None,
    predictions=True
)

import inspect
print(inspect.signature(LazyClassifier.fit))

clf_diag = LazyClassifier(
    verbose=1,
    ignore_warnings=False,
    custom_metric=None
)

modelos, previsoes = clf.fit(
    X_train,
    X_valid,
    y_train,
    y_valid
)
print("Comparação concluída.")

print(modelos.head(10))

print(modelos.shape)
print(modelos.columns.tolist())

metricas = ["Accuracy",
            "Balanced Accuracy",
            "ROC AUC", "F1 Score",
            "Time Taken"]

for m in metricas:
    print(m, m in modelos.columns)

if "ROC AUC" in modelos.columns:
    metrica_rank = "ROC AUC"
else:
    metrica_rank = "Balanced Accuracy"
    print("ROC AUC indisponivel nesta"
          " execução; usando Balanced Accuracy")

ranking = modelos.sort_values(
    by=metrica_rank, ascending=False
)

print("Métrica utilizada:", metrica_rank)
print(ranking.head(10))

linhas = [{"Referencia": "DummyClassifier (manual)",
           "Modelo": "Dummy",
           "Balanced Accuracy": bal_dummy}]

for m in ["Accuracy", "Balanced Accuracy",
          "ROC AUC"]:
    if m in modelos.columns:
        top = modelos[m].idxmax()
        linhas.append({"Referencia": "Top " + m,
                       "Modelo": top,
                       "Balanced Accuracy":
                       modelos.loc[top, "Balanced Accuracy"]})
comp = pd.DataFrame(linhas)
comp["Ganho vs Dummy"] = (
    comp["Balanced Accuracy"] - bal_dummy)

print(comp)

top10 = ranking.head(10)

fig, ax = plt.subplots(figsize=(9, 5))
barras = ax.barh(top10.index,
                 top10[metrica_rank],
                 color="#4158A6")
ax.invert_yaxis()
ax.bar_label(barras, fmt="%.4f", padding=4)
ax.set_xlabel(metrica_rank)
ax.set_title("Dez primeiros modelos por "
             + metrica_rank)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(figsize=(9, 5))
ax.scatter(modelos["Time Taken"],
           modelos[metrica_rank],
           color="#4158A6", s=50)

for nome, linha in modelos.iterrows():
    ax.annotate(nome, (linha["Time Taken"],
                       linha[metrica_rank]),
                       fontsize=7, color="#555B62")

ax.set_xlabel("Time Taken (s)")
ax.set_ylabel(metrica_rank)
ax.set_title("Desempenho vs. tempo")
plt.show()

print(type(previsoes))
print(previsoes.shape)
print(previsoes.head())

melhor_modelo = ranking.index[0]
print("Primeiro colocado:", melhor_modelo)

if melhor_modelo in previsoes.columns:
    y_pred_melhor = previsoes[melhor_modelo]
    print(classification_report(
        y_valid, y_pred_melhor,
        target_names=dados.target_names,
        zero_division=0
    ))
    cm_melhor = confusion_matrix(y_valid, y_pred_melhor, labels=[0, 1])
    print(pd.DataFrame(
        cm_melhor,
        index=["Real: malignant", "Real: benign"],
        columns=["Previsto: malignant", "Previsto: benign"]
    ))
else:
    print("As previsões desse modelo não foram encontradas.")

erros = getattr(clf, "errors", None)
print("Erros registrados: ", erros)

faltando = set(modelos.index) - set(
    getattr(previsoes, "columns", []))
print("Sem previsoes: ",faltando)

respostas = {
    1: f"Foram comparados {modelos.shape[0]} modelos.",
    2: f"{ranking.index[0]}, com {metrica_rank} = {ranking.iloc[0][metrica_rank]:.4f}.",
    3: f"{modelos['Balanced Accuracy'].idxmax()}, com Balanced Accuracy = {modelos['Balanced Accuracy'].max():.4f}.",
    4: f"{modelos['Time Taken'].idxmin()}, com {modelos['Time Taken'].min():.4f} s.",
    5: "Não. O ranking vale apenas para este conjunto de validação, esta métrica e estes dados; outro problema pode ter outro melhor modelo.",
    6: "Não. O LazyPredict treina os modelos com configurações padrão, sem busca sistemática de hiperparâmetros.",
    7: "Para comparar e escolher os modelos sem usar o teste, evitando que a escolha seja influenciada por ele.",
    8: "Quando já houver poucos modelos promissores e a validação indicar que ajustar hiperparâmetros pode trazer ganho.",
    9: "Para ter uma avaliação final independente do modelo escolhido, já que a validação foi usada na seleção."
}

for numero, resposta in respostas.items():
    print(f"{numero}. {resposta}")