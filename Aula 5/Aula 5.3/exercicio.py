import pandas as pd

from sklearn.datasets import load_diabetes

df = load_diabetes(as_frame=True, scaled=False).frame
df = df[["target", "age", "sex", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]]
df["target"] = df["target"].astype(int)

print(df.shape)

from sklearn.model_selection import train_test_split

X = df.drop(columns="target")
y = df["target"]

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=42
)


print(X_tr.isna().sum().sort_values(ascending=False))

num_cols = ["age", "bmi", "bp", "s1", "s2", "s3", "s4", "s5", "s6"]
cat_cols = ["sex"]

assert set(num_cols + cat_cols) == set(X_tr.columns)

print(set(num_cols + cat_cols) == set(X_tr.columns))

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

from sklearn.preprocessing import OneHotEncoder

cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

from sklearn.compose import ColumnTransformer

prep = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols)
])


from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ("prep", prep),
    ("reg", Ridge(alpha=1.0)),
])

scores = cross_val_score(pipe, X_tr, y_tr, cv=5, scoring="r2")

print(scores.mean().round(4), scores.std().round(4))

pipe.fit(X_tr, y_tr)

print(pipe.score(X_te, y_te))

names = pipe.named_steps["prep"].get_feature_names_out()
coofs = pipe.named_steps["reg"].coef_
pd.Series(coofs, index=names).sort_values()

print(pd.Series(coofs, index=names).sort_values())

from sklearn.model_selection import GridSearchCV

grid = {
    "prep__num__imputer__strategy": ["median", "mean"],
    "prep__cat__imputer__strategy": ["most_frequent", "constant"],
    "reg__alpha": [0.1, 1.0, 10.0, 100.0],
}
gs = GridSearchCV(pipe, grid, cv=5, scoring="r2", n_jobs=-1)

gs.fit(X_tr, y_tr)

gs.best_params_, round(gs.best_score_, 4)
print(gs.best_params_)
print(round(gs.best_score_, 4))

import joblib

joblib.dump(pipe, "diabetes_pipeline.joblib")
modelo = joblib.load("diabetes_pipeline.joblib")

nova = pd.DataFrame([{"age": 50, "sex": 1, "bmi": 25.0, "bp": 90.0,
                      "s1": 190.0, "s2": 110.0, "s3": 50.0,
                      "s4": 4.0, "s5": 4.5, "s6": 90.0}])

modelo.predict(nova).round(1)
print(modelo.predict(nova).round(1))