import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"

df = pd.read_csv(url)
y = df["Survived"]
X = df.drop(columns=["Survived", "PassengerId", "Ticket"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

class Features(BaseEstimator, TransformerMixin):
    def __init__(self, usar=()):
        self.usar = usar
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X = X.copy()
        X["FamilySize"] = X["SibSp"] + X["Parch"] + 1
        X["IsAlone"] = (X["FamilySize"] == 1).astype(int)
        X["FarePerPerson"] = X["Fare"] / X["FamilySize"]
        X["Title"] = (
            X["Name"]
            .str.extract(r",\s*([^\.]+)\.", expand=False)
            .str.strip()
            .replace(["Mlle", "Ms"], "Miss")
            .replace("Mme", "Mrs")
        )
        X["Title"] = X["Title"].where(
            X["Title"].isin(["Mr", "Miss", "Mrs", "Master"]),
            "Rare"
        )

        return X.drop(columns=["Name", "Cabin"])

def make_model(num_cols, cat_cols):
    num = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", RobustScaler())
    ])
    cat = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", min_frequency=10))
    ])
    pre = ColumnTransformer([
        ("num", num, num_cols),
        ("cat", cat, cat_cols)
    ])

    return Pipeline([
        ("feat", Features()),
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=1000))
    ])

NUM = ["Age", "SibSp", "Parch", "Fare"]
CAT = ["Pclass", "Sex", "Embarked"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

s = cross_val_score(
    make_model(NUM, CAT),
    X_train,
    y_train,
    cv=cv,
    scoring="accuracy"
)

ref = s.mean()

print(f"baseline {ref:.3f} +/- {s.std():.3f}")

LIMIAR = 0.002

candidatas = [
    ("FamilySize", "num"),
    ("IsAlone", "num"),
    ("FarePerPerson", "num"),
    ("Title", "cat")
]

num = list(NUM)
cat = list(CAT)

for nome, tipo in candidatas:

    if tipo == "num":
        n2, c2 = num + [nome], cat
    else:
        n2, c2 = num, cat + [nome]
    s = cross_val_score(
        make_model(n2, c2),
        X_train,
        y_train,
        cv=cv,
        scoring="accuracy"
    )
    ganho = s.mean() - ref
    print(
        f"{nome:>14} {s.mean():.3f} +/- {s.std():.3f} ganho {ganho:+.3f}"
    )

    if ganho > LIMIAR:
        num, cat, ref = n2, c2, s.mean()
    
print("features mantidas:", num, cat)
final = make_model(num, cat).fit(X_train, y_train)
print(f"acuracia no teste: {final.score(X_test, y_test):.3f}")
