import sys, pandas as pd, seaborn as sns, pycaret
from pycaret.classification import *
print(sys.version.split()[0], pycaret.__version__)

df = sns.load_dataset("titanic")
df = df[("survived", "pclass", "sex", "age", 
         "sibsp", "parch", "fare", "embarked")]
print(df.shape)
print(df.isna().sum())
print(1 - df["survived"].mean())