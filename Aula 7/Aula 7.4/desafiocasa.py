import time
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from catboost import CatBoostClassifier

np.set_printoptions(precision=4, suppress=True)

print("\n BIBLIOTECAS E VERSÕES")

pacotes = [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "xgboost",
    "lightgbm",
    "catboost",
]

versoes = {}
for pacote in pacotes:
    versoes[pacote] = importlib.metadata.version(pacote)
    print(f"{pacote}: {versoes[pacote]}")

print("\n CARREGAMENTO DO DATASET")

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

print("Formato de X:", X.shape)

print("Formato de y:", y.shape)

print("Classes e contagens:", np.unique(y, return_counts=True))

print("\n EXPLORAÇÃO DOS DADOS")

colunas = list(X.columns)
df = X.copy()
df["target"] = y

print("\nPrimeiras linhas:")

print(df.head())

print("\nDimensões do DataFrame:", df.shape)

print("\nTipos de dados:")

print(df.dtypes.value_counts())

print("\nDistribuição das classes:")

print(df["target"].value_counts().sort_index())

print("\nDistribuição proporcional:")

print(df["target"].value_counts(normalize=True).sort_index())

print("\nEstatísticas descritivas:")

print(df.describe().T)

fig, ax = plt.subplots(figsize=(6, 4))
contagens = df["target"].value_counts().sort_index()
ax.bar(["Classe 0", "Classe 1"], contagens.values,
       color=["#75BCE8", "#62CDB5"])
ax.set_title("Distribuição dos rótulos")
ax.set_ylabel("Quantidade de amostras")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
plt.show()

print("\n TREINO, VALIDAÇÃO E TESTE")

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42,
)

X_train, X_val, y_train, y_val = train_test_split(
    X_temp,
    y_temp,
    test_size=0.25,
    stratify=y_temp,
    random_state=42,
)

print("Treino:", X_train.shape, y_train.shape)

print("Validação:", X_val.shape, y_val.shape)

print("Teste:", X_test.shape, y_test.shape)

print("\n BASELINE")

baseline = DummyClassifier(strategy="prior")

inicio = time.perf_counter()
baseline.fit(X_train, y_train)
tempo_baseline = time.perf_counter() - inicio

pred_base = baseline.predict(X_test)
acc_base = accuracy_score(y_test, pred_base)

print(f"Acurácia do baseline: {acc_base:.4f}")

print(f"Tempo de treinamento do baseline: {tempo_baseline:.4f} s")

print("\n XGBOOST")

modelo_xgb = XGBClassifier(
    n_estimators=800,
    max_depth=5,
    learning_rate=0.05,
    objective="binary:logistic",
    eval_metric="auc",
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    early_stopping_rounds=40,
    random_state=42,
    tree_method="hist",
    n_jobs=-1,
)

inicio = time.perf_counter()
modelo_xgb.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
tempo_xgb = time.perf_counter() - inicio

best_iteration_xgb = modelo_xgb.best_iteration
arvores_xgb = int(best_iteration_xgb) + 1

print(f"Tempo medido do XGBoost: {tempo_xgb:.4f} s")

print("Melhor iteração do XGBoost:", best_iteration_xgb)

print("Árvores efetivamente utilizadas:", arvores_xgb)

print("\n LIGHTGBM")

modelo_lgbm = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=5,
    num_leaves=31,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    objective="binary",
    metric="auc",
    random_state=42,
    n_jobs=-1,
    verbosity=-1,
)

inicio = time.perf_counter()
modelo_lgbm.fit(
    X_train,
    y_train,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    callbacks=[
        early_stopping(40, verbose=False),
        log_evaluation(0),
    ],
)
tempo_lgbm = time.perf_counter() - inicio

best_iteration_lgbm = modelo_lgbm.best_iteration_
if best_iteration_lgbm in (None, 0):
    arvores_lgbm = modelo_lgbm.booster_.current_iteration()
else:
    arvores_lgbm = int(best_iteration_lgbm)

print(f"Tempo medido do LightGBM: {tempo_lgbm:.4f} s")

print("Melhor iteração do LightGBM:", best_iteration_lgbm)

print("Árvores efetivamente utilizadas:", arvores_lgbm)

print("\n CATBOOST")

modelo_cat = CatBoostClassifier(
    iterations=800,
    depth=5,
    learning_rate=0.05,
    loss_function="Logloss",
    eval_metric="AUC",
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=False,
    allow_writing_files=False,
    thread_count=-1,
)

inicio = time.perf_counter()
modelo_cat.fit(
    X_train,
    y_train,
    eval_set=(X_val, y_val),
    early_stopping_rounds=40,
    verbose=False,
)
tempo_cat = time.perf_counter() - inicio

arvores_cat = modelo_cat.tree_count_
best_iteration_cat = modelo_cat.best_iteration_

print(f"Tempo medido do CatBoost: {tempo_cat:.4f} s")

print("Melhor iteração do CatBoost:", best_iteration_cat)

print("Árvores efetivamente utilizadas:", arvores_cat)

print("\n CONFIGURAÇÕES UTILIZADAS")

df_config = pd.DataFrame([
    {"Modelo": "XGBoost", "n_estimators": 800, "learning_rate": 0.05,
     "profundidade": 5, "regularização L2": 1.0, "árvores usadas": arvores_xgb},
    {"Modelo": "LightGBM", "n_estimators": 800, "learning_rate": 0.05,
     "profundidade": 5, "regularização L2": 1.0, "árvores usadas": arvores_lgbm},
    {"Modelo": "CatBoost", "n_estimators": 800, "learning_rate": 0.05,
     "profundidade": 5, "regularização L2": 3.0, "árvores usadas": arvores_cat},
])

print(df_config.to_string(index=False))

print("\n CONFERÊNCIA DA COMPARAÇÃO JUSTA")

assert set(X_train.index).isdisjoint(set(X_val.index))
assert set(X_train.index).isdisjoint(set(X_test.index))
assert set(X_val.index).isdisjoint(set(X_test.index))
assert set(np.unique(y_train)) == {0, 1}
assert set(np.unique(y_val)) == {0, 1}
assert set(np.unique(y_test)) == {0, 1}

print("OK: mesmo dataset, mesmos conjuntos, mesmos rótulos e mesma métrica de avaliação (AUC).")

print("Observação: os hiperparâmetros internos não são perfeitamente equivalentes entre as bibliotecas.")

print("Este benchmark compara configurações documentadas sob o mesmo protocolo; não prova superioridade universal.")

print("\n PROBABILIDADES E LIMIAR COMUM")


def probabilidade_classe_positiva(modelo, X_entrada):
    probabilidades = modelo.predict_proba(X_entrada)
    indice_positivo = list(modelo.classes_).index(1)
    return probabilidades[:, indice_positivo]


LIMIAR = 0.5
modelos = {
    "XGBoost": modelo_xgb,
    "LightGBM": modelo_lgbm,
    "CatBoost": modelo_cat,
}

tempos = {
    "Baseline": tempo_baseline,
    "XGBoost": tempo_xgb,
    "LightGBM": tempo_lgbm,
    "CatBoost": tempo_cat,
}

proba_teste = {
    "Baseline": probabilidade_classe_positiva(baseline, X_test),
}
pred_teste = {
    "Baseline": (proba_teste["Baseline"] >= LIMIAR).astype(int),
}

for nome, modelo in modelos.items():
    p = probabilidade_classe_positiva(modelo, X_test)
    proba_teste[nome] = p
    pred_teste[nome] = (p >= LIMIAR).astype(int)

print("Probabilidades e previsões calculadas para todos os modelos.")

print("Limiar comum:", LIMIAR)

print("\n MÉTRICAS COMUNS")


def calcular_metricas(nome, y_true, proba, tempo=np.nan):
    pred = (proba >= LIMIAR).astype(int)
    return {
        "Modelo": nome,
        "Acurácia": accuracy_score(y_true, pred),
        "Precisão": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "ROC AUC": roc_auc_score(y_true, proba),
        "Log Loss": log_loss(y_true, proba, labels=[0, 1]),
        "Tempo (s)": tempo,
    }


resultados = []
for nome in ["Baseline", "XGBoost", "LightGBM", "CatBoost"]:
    resultados.append(
        calcular_metricas(nome, y_test, proba_teste[nome], tempos[nome])
    )

df_metricas = pd.DataFrame(resultados)

print(df_metricas.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))

print("\n MATRIZES DE CONFUSÃO")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, nome in zip(axes, ["XGBoost", "LightGBM", "CatBoost"]):
    cm = confusion_matrix(y_test, pred_teste[nome], labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=[0, 1])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(nome)
fig.suptitle("Matrizes de confusão no conjunto de teste")
fig.tight_layout()
plt.show()

print("\n TABELA COMPARATIVA NO TESTE")

print("\nVersões utilizadas:")
for pacote, versao in versoes.items():
    print(f"{pacote}: {versao}")

print("\nTabela final:")

print(df_metricas.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))

print("\n TEMPO DE TREINAMENTO")

nomes_tempo = ["XGBoost", "LightGBM", "CatBoost"]
valores_tempo = [tempo_xgb, tempo_lgbm, tempo_cat]

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(nomes_tempo, valores_tempo,
       color=["#75BCE8", "#62CDB5", "#F4CA70"])
ax.set_title("Tempo medido de treinamento")
ax.set_ylabel("Segundos")
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()
plt.show()

print("Tempos de treinamento (s):")
for nome, valor in zip(nomes_tempo, valores_tempo):
    print(f"{nome}: {valor:.4f}")

print("Árvores/iterações efetivamente utilizadas:")

print({
    "XGBoost": arvores_xgb,
    "LightGBM": arvores_lgbm,
    "CatBoost": arvores_cat,
})

print("\n IMPORTÂNCIA POR PERMUTAÇÃO")

def auc_validacao(modelo, X_avaliacao, y_avaliacao):
    proba = probabilidade_classe_positiva(modelo, X_avaliacao)
    return roc_auc_score(y_avaliacao, proba)


importancias = {}
for nome, modelo in modelos.items():
    resultado = permutation_importance(
        modelo,
        X_val,
        y_val,
        scoring=auc_validacao,
        n_repeats=5,
        random_state=42,
        n_jobs=1,
    )
    importancias[nome] = resultado.importances_mean

    print(f"\n{nome} — maiores importâncias por permutação:")

    ranking = pd.Series(resultado.importances_mean, index=colunas)

    print(ranking.sort_values(ascending=False).head(10))

fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=False)
for ax, nome in zip(axes, ["XGBoost", "LightGBM", "CatBoost"]):
    ranking = pd.Series(importancias[nome], index=colunas).sort_values().tail(10)
    ax.barh(ranking.index, ranking.values, color="#75BCE8")
    ax.set_title(nome)
    ax.set_xlabel("Queda média de ROC AUC após permutação")
fig.suptitle("Importância por permutação na validação")
fig.tight_layout()
plt.show()

print("Observação: a importância não significa causalidade. A validação também foi usada para early stopping; portanto, esta análise é exploratória.")

print("\n CONCLUSÃO DO BENCHMARK")

df_boosting = df_metricas[df_metricas["Modelo"].isin(modelos.keys())].copy()
melhor_auc = df_boosting.loc[df_boosting["ROC AUC"].idxmax(), "Modelo"]
melhor_f1 = df_boosting.loc[df_boosting["F1"].idxmax(), "Modelo"]
menor_logloss = df_boosting.loc[df_boosting["Log Loss"].idxmin(), "Modelo"]
menor_tempo = df_boosting.loc[df_boosting["Tempo (s)"].idxmin(), "Modelo"]

print("Melhor ROC AUC entre os três modelos:", melhor_auc)

print("Melhor F1 entre os três modelos:", melhor_f1)

print("Menor Log Loss entre os três modelos:", menor_logloss)

print("Menor tempo de treinamento entre os três modelos:", menor_tempo)

linhas_gap = []
for nome, modelo in modelos.items():
    auc_treino = roc_auc_score(
        y_train, probabilidade_classe_positiva(modelo, X_train)
    )
    auc_val = roc_auc_score(
        y_val, probabilidade_classe_positiva(modelo, X_val)
    )
    linhas_gap.append({
        "Modelo": nome,
        "ROC AUC treino": auc_treino,
        "ROC AUC validação": auc_val,
        "Gap treino - validação": auc_treino - auc_val,
    })

df_gap = pd.DataFrame(linhas_gap)

print("\nGap de ROC AUC entre treino e validação:")

print(df_gap.to_string(index=False, float_format=lambda valor: f"{valor:.4f}"))

df_importancias = pd.DataFrame(importancias, index=colunas)
correlacao_importancias = df_importancias.corr(method="spearman")

print("\nCorrelação de Spearman entre rankings de importância:")

print(correlacao_importancias.to_string(float_format=lambda valor: f"{valor:.3f}"))

print("\nLimitações do experimento:")

print("- Dataset pequeno (569 amostras) e um único split.")

print("- Hiperparâmetros específicos, não equivalentes internamente.")

print("- Tempos de uma execução nesta máquina.")

print("- Pequenas diferenças podem mudar com outra semente ou hardware.")

print("- Este resultado não comprova superioridade universal de uma biblioteca.")

resumo = pd.DataFrame({
    "Critério": ["Melhor ROC AUC", "Melhor F1", "Menor Log Loss", "Menor tempo entre Boosting"],
    "Modelo": [melhor_auc, melhor_f1, menor_logloss, menor_tempo],
})

print(resumo.to_string(index=False))