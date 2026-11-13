import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_diabetes
from sklearn.model_selection import (train_test_split, KFold, cross_val_score, 
                                     GridSearchCV)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (LinearRegression, Ridge,
                                  Lasso, ElasticNet)
from sklearn.metrics import mean_squared_error, r2_score

SEED = 42
print("sklearn", sklearn.__version__)
X, y = load_diabetes(return_X_y=True, as_frame=True)
print(X.shape)
print(X.corrwith(y).sort_values())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
SCORING = "neg_root_mean_squared_error"

def make_pipe(model):
    return Pipeline([("scaler", StandardScaler()),
                     ("model", model)])

baseline = make_pipe(LinearRegression())
scores = cross_val_score(baseline, X_train, y_train,
                         cv=cv, scoring=SCORING)

print(f"MQD RMSE-CV = {-scores.mean():.2f} ± {scores.std():.2f}")

ridge_search = GridSearchCV(
    make_pipe(Ridge()),
    {"model__alpha": np.logspace(-3, 3, 61)},
    cv=cv, scoring=SCORING)
ridge_search.fit(X_train, y_train)

lasso_search = GridSearchCV(
    make_pipe(Lasso(max_iter=50_000)),
    {"model__alpha": np.logspace(-3, 1.5, 56)},
    cv=cv, scoring=SCORING)
lasso_search.fit(X_train, y_train)

print(ridge_search.best_params_, -ridge_search.best_score_)
print(lasso_search.best_params_, -lasso_search.best_score_)

enet_search = GridSearchCV(
    make_pipe(ElasticNet(max_iter=50_000)),
    {"model_alpha": np.logspace(-3, 1.5, 31),
     "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]},
     cv=cv, scoring=SCORING)

resumo = pd.DataFrame({
    "modelo": ["MQO", "RIDGE", "LASSO", "ELASTICNET"],
    "RMSE_CV": [-scores.mean(), -ridge_search.best_score_,
                -lasso_search.best_score_,
                -enet_search.best_score_]})

print(resumo.sort_values("RMSE_CV"))