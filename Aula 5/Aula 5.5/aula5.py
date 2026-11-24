from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (
    SelectKBest,
    f_regression,
    mutual_info_regression,
    RFE,
    SelectFromModel
)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
import numpy as np
import features


# carrega os dados
X, y = fetch_california_housing(
    return_X_y=True,
    as_frame=True
)

# divide treino e teste
X_tr, X_te, y_tr, y_te = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Tamanho do treino:", X_tr.shape)
print("Tamanho do teste:", X_te.shape)


# validação cruzada
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# modelo base
modelo = HistGradientBoostingRegressor(
    random_state=42
)

base = cross_val_score(
    modelo,
    X_tr,
    y_tr,
    cv=cv,
    scoring="r2"
)

print("Modelo base:")
print("Média:", base.mean())
print("Desvio padrão:", base.std())


# selectKBest
pipe = Pipeline([
    ("sel", SelectKBest(f_regression, k=4)),
    ("modelo", modelo),
])

sel = cross_val_score(
    pipe,
    X_tr,
    y_tr,
    cv=cv,
    scoring="r2"
)

print("\nSelectKBest:")
print("Média:", sel.mean())
print("Desvio padrão:", sel.std())


# mutual information
pipe_mi = Pipeline([
    ("sel", SelectKBest(mutual_info_regression, k=4)),
    ("modelo", modelo),
])

mi = cross_val_score(
    pipe_mi,
    X_tr,
    y_tr,
    cv=cv,
    scoring="r2"
)

print("\nMutual Information:")
print("Média:", mi.mean())
print("Desvio padrão:", mi.std())


# rfe
pipe_rfe = Pipeline([
    ("sc", StandardScaler()),
    ("sel", RFE(
        LinearRegression(),
        n_features_to_select=4
    )),
    ("modelo", modelo),
])

rfe = cross_val_score(
    pipe_rfe,
    X_tr,
    y_tr,
    cv=cv,
    scoring="r2"
)

print("\nRFE:")
print("Média:", rfe.mean())
print("Desvio padrão:", rfe.std())


# selectfrommodel com lasso
pipe_lasso = Pipeline([
    ("sc", StandardScaler()),
    ("sel", SelectFromModel(
        LassoCV(random_state=42)
    )),
    ("modelo", modelo),
])

lasso = cross_val_score(
    pipe_lasso,
    X_tr,
    y_tr,
    cv=cv,
    scoring="r2"
)

print("\nLasso:")
print("Média:", lasso.mean())
print("Desvio padrão:", lasso.std())


# SelectFromModel com Random Forest
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

pipe_rf = Pipeline([
    ("sel", SelectFromModel(
        rf,
        threshold=-np.inf,
        max_features=4
    )),
    ("modelo", modelo),
])

rf_score = cross_val_score(
    pipe_rf,
    X_tr,
    y_tr,
    cv=cv,
    scoring="r2"
)

print("\nRandom Forest:")
print("Média:", rf_score.mean())
print("Desvio padrão:", rf_score.std())


# ajusta o SelectKBest no conjunto de treino
pipe.fit(X_tr, y_tr)

print("\nFeatures selecionadas:")

features_selecionadas = pipe[:-1].get_feature_names_out()

print(features_selecionadas)

# avaliacao no conjunto de teste
print("\nR² no teste:")
print(pipe.score(X_te, y_te))