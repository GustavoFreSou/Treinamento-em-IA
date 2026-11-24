import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.trial import TrialState
from scipy.stats import randint, loguniform
from sklearn.datasets import load_wine
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV,
    RandomizedSearchCV
)
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    confusion_matrix
)

print("NumPy:", np.__version__)

print("Pandas:", pd.__version__)

print("Optuna:", optuna.__version__)

import scipy, sklearn

print("SciPy:", scipy.__version__)

print("Scikit-learn:", sklearn.__version__)

wine = load_wine()
X, y = wine.data, wine.target

print("Formato de X:", X.shape)

print("Formato de y:", y.shape)

print("Classes e contagens:", np.unique(y, return_counts=True))

print("Nomes das classes:", list(wine.target_names))

df = pd.DataFrame(X, columns=wine.feature_names)
df["classe"] = y

print("Primeiras linhas:")

print(df.head())

print("\nEstatísticas descritivas:")

print(df.describe())

print("\nProporção das classes:")

print(df["classe"].value_counts(normalize=True).sort_index())

contagens = df["classe"].value_counts().sort_index()
plt.figure(figsize=(6, 4))
plt.bar(contagens.index.astype(str), contagens.values)
plt.title("Distribuição das classes")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.tight_layout()
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Treino:", X_train.shape, y_train.shape)

print("Teste:", X_test.shape, y_test.shape)

print("Distribuição no treino:", np.unique(y_train, return_counts=True))

print("Distribuição no teste:", np.unique(y_test, return_counts=True))

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)
splits = list(cv.split(X_train, y_train))

print("Número de folds:", len(splits))
for i, (idx_tr, idx_va) in enumerate(splits, start=1):
    print(f"Fold {i}: ajuste={len(idx_tr)}, validação={len(idx_va)}")

baseline = GradientBoostingClassifier(
    n_estimators=150,
    random_state=42
)
acc_baseline = cross_val_score(
    baseline,
    X_train,
    y_train,
    cv=splits,
    scoring="accuracy",
    n_jobs=1
).mean()

print(f"Acurácia média do baseline: {acc_baseline:.4f}")

param_grid = {
    "max_depth": [1, 2, 3],
    "learning_rate": [0.03, 0.10],
    "min_samples_leaf": [1, 5]
}

n_config_grid = int(np.prod([len(v) for v in param_grid.values()]))

print("Configurações Grid:", n_config_grid)

print("Ajustes de modelo (configurações × folds):", n_config_grid * len(splits))

modelo_base = GradientBoostingClassifier(
    n_estimators=150,
    random_state=42
)

grid = GridSearchCV(
    estimator=modelo_base,
    param_grid=param_grid,
    scoring="accuracy",
    cv=splits,
    n_jobs=1,
    refit=False,
    return_train_score=True,
    error_score="raise"
)

inicio = time.perf_counter()
grid.fit(X_train, y_train)
tempo_grid = time.perf_counter() - inicio

print("Melhores parâmetros:", grid.best_params_)

print(f"Melhor acurácia média (CV): {grid.best_score_:.4f}")

print(f"Tempo de busca: {tempo_grid:.2f} s")

resultados_grid = pd.DataFrame(grid.cv_results_)
colunas_grid = [
    "params", "mean_test_score", "std_test_score",
    "mean_train_score", "rank_test_score", "mean_fit_time"
]

print(resultados_grid[colunas_grid].sort_values("rank_test_score").to_string(index=False))

param_distributions = {
    "max_depth": randint(1, 4),
    "learning_rate": loguniform(0.02, 0.15),
    "min_samples_leaf": randint(1, 6)
}

random_search = RandomizedSearchCV(
    estimator=modelo_base,
    param_distributions=param_distributions,
    n_iter=12,
    scoring="accuracy",
    cv=splits,
    random_state=42,
    n_jobs=1,
    refit=False,
    return_train_score=True,
    error_score="raise"
)

inicio = time.perf_counter()
random_search.fit(X_train, y_train)
tempo_random = time.perf_counter() - inicio

print("Melhores parâmetros:", random_search.best_params_)

print(f"Melhor acurácia média (CV): {random_search.best_score_:.4f}")

print(f"Tempo de busca: {tempo_random:.2f} s")

resultados_random = pd.DataFrame(random_search.cv_results_)
colunas_random = [
    "params", "mean_test_score", "std_test_score",
    "mean_train_score", "rank_test_score", "mean_fit_time"
]

print(resultados_random[colunas_random].sort_values("rank_test_score").to_string(index=False))

sampler = optuna.samplers.TPESampler(
    seed=42,
    n_startup_trials=5
)

pruner = optuna.pruners.MedianPruner(
    n_startup_trials=5,
    n_warmup_steps=50,
    interval_steps=25,
    n_min_trials=3
)

CHECKPOINTS = [25, 50, 75, 100, 150]
N_FOLDS = len(splits)
N_CLASSES = len(np.unique(y))

print("Checkpoints:", CHECKPOINTS)

print("Folds usados em cada checkpoint:", N_FOLDS)

def objective(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 1, 3),
        "learning_rate": trial.suggest_float(
            "learning_rate", 0.02, 0.15, log=True
        ),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5)
    }

    modelos = [
        GradientBoostingClassifier(
            n_estimators=CHECKPOINTS[0],
            warm_start=True,
            random_state=42,
            **params
        )
        for _ in splits
    ]

    media_acc = None

    for checkpoint in CHECKPOINTS:
        accs = []

        for modelo, (idx_tr, idx_va) in zip(modelos, splits):
            modelo.set_params(n_estimators=checkpoint)
            modelo.fit(X_train[idx_tr], y_train[idx_tr])

            previsoes = modelo.predict(X_train[idx_va])
            accs.append(accuracy_score(y_train[idx_va], previsoes))

        media_acc = float(np.mean(accs))
        trial.set_user_attr("ultimo_checkpoint", checkpoint)
        trial.report(media_acc, step=checkpoint)

        if checkpoint < CHECKPOINTS[-1] and trial.should_prune():
            raise optuna.TrialPruned()

    return media_acc

print("Função objetivo definida.")

study = optuna.create_study(
    direction="maximize",
    sampler=sampler,
    pruner=pruner
)

inicio = time.perf_counter()
study.optimize(
    objective,
    n_trials=12,
    n_jobs=1,
    show_progress_bar=True
)
tempo_optuna = time.perf_counter() - inicio

completos = study.get_trials(states=[TrialState.COMPLETE])
podados = study.get_trials(states=[TrialState.PRUNED])

print("Melhores parâmetros:", study.best_params)

print(f"Melhor acurácia média (CV): {study.best_value:.4f}")

print("Trials completos:", len(completos))

print("Trials interrompidos:", len(podados))

print(f"Tempo de busca: {tempo_optuna:.2f} s")

linhas_trials = []
for t in study.trials:
    linhas_trials.append({
        "trial": t.number,
        "estado": t.state.name,
        "acuracia_final": t.value,
        "ultimo_checkpoint": t.user_attrs.get("ultimo_checkpoint", 0),
        **t.params
    })

df_trials = pd.DataFrame(linhas_trials)

print(df_trials.to_string(index=False))

arvores_grid = n_config_grid * N_FOLDS * 150 * N_CLASSES
arvores_random = random_search.n_iter * N_FOLDS * 150 * N_CLASSES
arvores_optuna = sum(
    t.user_attrs.get("ultimo_checkpoint", 0) * N_FOLDS * N_CLASSES
    for t in study.trials
)

orcamento = pd.DataFrame({
    "Estratégia": ["Grid Search", "Random Search", "Optuna"],
    "Configurações/trials": [n_config_grid, random_search.n_iter, len(study.trials)],
    "Trials interrompidos": [0, 0, len(podados)],
    "Árvores-base (aprox.)": [arvores_grid, arvores_random, arvores_optuna],
    "Tempo de busca (s)": [tempo_grid, tempo_random, tempo_optuna]
})

print(orcamento.to_string(index=False))

melhores = {
    "Grid Search": grid.best_params_,
    "Random Search": random_search.best_params_,
    "Optuna (TPE + pruning)": study.best_params
}

scores_cv = {
    "Grid Search": grid.best_score_,
    "Random Search": random_search.best_score_,
    "Optuna (TPE + pruning)": study.best_value
}

tempos_busca = {
    "Grid Search": tempo_grid,
    "Random Search": tempo_random,
    "Optuna (TPE + pruning)": tempo_optuna
}

modelos_finais = {}
tempos_fit_final = {}

for nome, params in melhores.items():
    modelo_final = GradientBoostingClassifier(
        n_estimators=150,
        random_state=42,
        **params
    )
    inicio = time.perf_counter()
    modelo_final.fit(X_train, y_train)
    tempos_fit_final[nome] = time.perf_counter() - inicio
    modelos_finais[nome] = modelo_final

print("Configurações selecionadas:")

for nome, params in melhores.items():
    print(f"{nome}: {params}")

print("\nTempos de ajuste final:", tempos_fit_final)

resultados_teste = {}
matrizes_confusao = {}

for nome, modelo in modelos_finais.items():
    proba = modelo.predict_proba(X_test)
    pred = modelo.predict(X_test)

    resultados_teste[nome] = {
        "Acurácia teste": accuracy_score(y_test, pred),
        "F1 macro": f1_score(y_test, pred, average="macro"),
        "Log Loss": log_loss(y_test, proba, labels=[0, 1, 2])
    }
    matrizes_confusao[nome] = confusion_matrix(y_test, pred, labels=[0, 1, 2])

    print(f"\n{nome}")

    for metrica, valor in resultados_teste[nome].items():
        print(f"{metrica}: {valor:.4f}")

    print("Matriz de confusão (linhas = real; colunas = previsto):")

    print(matrizes_confusao[nome])

linhas = []
for nome in melhores:
    linhas.append({
        "Estratégia": nome,
        "Acurácia CV": scores_cv[nome],
        "Acurácia teste": resultados_teste[nome]["Acurácia teste"],
        "F1 macro": resultados_teste[nome]["F1 macro"],
        "Log Loss": resultados_teste[nome]["Log Loss"],
        "Busca (s)": tempos_busca[nome],
        "Ajuste final (s)": tempos_fit_final[nome]
    })

tabela_final = pd.DataFrame(linhas).set_index("Estratégia")

print(tabela_final.round(4).to_string())

plt.figure(figsize=(8, 4.5))
plt.bar(tabela_final.index, tabela_final["Acurácia teste"])
plt.ylabel("Acurácia no teste")
plt.title("Comparação final — Acurácia")
plt.ylim(0, 1)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4.5))
plt.bar(tabela_final.index, tabela_final["Busca (s)"])
plt.ylabel("Segundos")
plt.title("Tempo de busca por estratégia")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.show()

trial_nums = [t.number for t in study.trials]
trial_scores = [np.nan if t.value is None else t.value for t in study.trials]
trial_colors = ["#EFA6B8" if t.state == TrialState.PRUNED else "#62CDB5" for t in study.trials]

plt.figure(figsize=(8, 4.5))
plt.scatter(trial_nums, trial_scores, c=trial_colors, s=80)
plt.xlabel("Número do trial")
plt.ylabel("Acurácia final (somente trials completos)")
plt.title("Histórico de trials do Optuna")
plt.grid(alpha=0.25)
plt.tight_layout()
plt.show()

print("Legenda: verde-menta = completo; rosa = interrompido.")

fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
for ax, (nome, cm) in zip(axes, matrizes_confusao.items()):
    ax.imshow(cm, cmap="Blues")
    ax.set_title(nome, fontsize=9)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    for (i, j), valor in np.ndenumerate(cm):
        ax.text(j, i, str(valor), ha="center", va="center")
fig.suptitle("Matrizes de confusão — teste")
plt.tight_layout()
plt.show()
