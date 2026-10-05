import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
modelo = HistGradientBoostingRegressor(random_state=42)

# teste com as 8 features 
modelo.fit(X_tr, y_tr)
r2_todas = modelo.score(X_te, y_te)
print("Features antes:", X.shape[1], "| R² teste:", round(r2_todas, 4))

# com validação cruzada treina para ver a influencia da aplicacao dos features
for k in range(1, 9):
    pipe_k = Pipeline([("sel", SelectKBest(f_regression, k=k)),
                       ("modelo", HistGradientBoostingRegressor(random_state=42))])
    s = cross_val_score(pipe_k, X_tr, y_tr, cv=cv, scoring="r2")
    print(f"k={k}: R² CV = {s.mean():.4f} ± {s.std():.4f}")

# escolhe melhor modelo, baseado nos testes de cima
K = 6
pipe = Pipeline([("sel", SelectKBest(f_regression, k=K)),
                 ("modelo", HistGradientBoostingRegressor(random_state=42))])
pipe.fit(X_tr, y_tr)
r2_sel = pipe.score(X_te, y_te)

mantidas = list(pipe[:-1].get_feature_names_out())
removidas = [c for c in X.columns if c not in mantidas]
print("Features depois:", len(mantidas), "| R² teste:", round(r2_sel, 4))
print("Mantidas:", mantidas)
print("Removidas:", removidas)



# comparei dois modelos com o mesmo algoritmo (histgradientboosting) e a mesma
# divisão treino/teste. um usa as 8 features e o outro
# usa só as selecionadas com selectkbest (f_regression). coloquei a seleção
# dentro do pipeline pra ela ser refeita a cada fold da validação cruzada e
# não vazar informação do teste
