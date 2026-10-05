from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

X, y = fetch_california_housing( 
    return_X_y=True, as_frame=True)

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42)

X_tr.shape

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

cv = KFold(n_split=5, shuffle=True, random_state=42)
modelo = HistGradientBoostingRegressor(random_state=42)

base = cross_val_score(modelo, X_tr, y_tr,
                       cv=cv, scoring="r2")
base.mean(), base.std()

print(base.mean(), base.std())

from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (SelectKBest, f_regression)

pipe = Pipeline([
    ("sel", SelectKBest(f_regression, k=4)),
    ("modelo", modelo),
])
sel = cross_val_score(pipe, X_tr, y_tr,
                      cv=cv, scoring="r2")
sel.mean(), sel.std()

("sel", SelectKBest(mi, k=4))
("sc", StandardScaler()), ("sel", RFE(LinearRegression(), n_features_to_select=6))
("sc", StandardScaler()), ("sel", SelectFromModel(LassoCV(random_state=42)))
("sel", SelectFromModel(rf, threshold=-np.inf, max_features=6))

pipe.fit(X_tr, y_tr)
pipe[:-1].get_feature_names_out()
pipe.score(X_te, y_te)