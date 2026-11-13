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

baseline = DummyClassifier(strategy="prior")
baseline.fit(X_train, y_train)
pred_baseline = baseline.predict(X_valid)
acc_dummy = accuracy_score(y_valid, pred_baseline)
bal_dummy = balanced_accuracy_score(
    y_valid, pred_baseline
)
print(f"Dummy * Accuracy:        {acc_dummy:.4f}")
print(f"Dummy * Balanced Accuracy:        {bal_dummy:.4f}")

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
    metrica_rank = "ROC AUC"
    print("ROC AUC indisponivel nest"
          " execução; usando Balanced Accuracy")

ranking = modelos.sort_values(
    by=metrica_rank, ascending=False
)

print(ranking.head(10))

linhas = [{"referencia": "DummyClassifier (manual)",
           "Modelo": "Dummy",
           "Balanced Accuracy": bal_dummy}]

for m in ["Accuracy", "Balanced Accuracy",
          "ROC AUC"]:
    if m in modelos.columns:
        top = modelos[m].idxmax()
        linhas.append({"Referencia": "Top" + m, 
                        "modelo": top,
                        "Balanced Accuracy":
                        modelos.loc[top, "balanced Accuracy"]})
comp = pd.DataFrame(linhas)
comp["Ganho vs Dummy"] = (
    comp["Balanced Accuracy"] - bal_dummy)

print(comp)

