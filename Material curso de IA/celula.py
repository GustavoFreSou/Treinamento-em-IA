import pandas as pd

from sklearn.datasets import load_iris

iris = load_iris

df = pd.DataFrame(x, columns=iris.feature_names)

df['especie'] = iris.target_names[y]

df.groupby('especie').mean().round(2)