import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    ConfusionMatrixDisplay
)

df = sns.load_dataset("penguins")
df.head()
df.columns
df.info()
df.isnull().sum()
df.describe()

plt.bar(df['species'], df['island'])
plt.xlabel('species')
plt.ylabel('island')
plt.title('species x island')
plt.show()

print("shape:", df.shape)

X = df.drop(columns= "species")
y = df["species"]


num_cols = ["body_mass_g", "flipper_length_mm", "bill_length_mm", "bill_depth_mm"]
cat_cols = ["sex", "island", "species"]

X = X[num_cols + cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state= 42,

)

print(X_train.shape, X_test.shape)

num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most-frequent")),
    ("Onehot", OneHotEncoder(handle_unknown="ignore")),
])
Preprocessor = ColumnTransformer([
    ("num", num_pipe, num_cols )
    ("cat", cat_pipe , cat_cols )
])

modelo= LogisticRegression(max_iter=1000)
Pipe = Pipeline([
    "prepocessor, preprocessor",
    "model, modelo"
])

Pipe.fit(X_train, y_train)
print(Pipe.fit(X_train, y_train))