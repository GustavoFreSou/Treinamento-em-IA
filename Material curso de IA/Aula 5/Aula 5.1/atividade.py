from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

iris = load_iris()

df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df['target'] = iris.target

df['species'] = df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

df.head()

print("Shape:", df.shape)
print("\nColunas:", list(df.columns))
df.info()

print("Valores ausentes por coluna:")
print(df.isnull().sum())

print("\nQuantidade de duplicatas:", df.duplicated().sum())

df.describe()
df['species'].value_counts()
df.groupby('species').mean(numeric_only=True)

import matplotlib.pyplot as plt

df['species'].value_counts().plot(kind='bar', color=['#4C72B0', '#DD8452', '#55A868'])
plt.title('Quantidade de amostras por espécie')
plt.xlabel('Espécie')
plt.ylabel('Quantidade')
plt.show()

df['petal length (cm)'].plot(kind='hist', bins=20, color='teal', edgecolor='black')
plt.title('Distribuição do comprimento da pétala')
plt.xlabel('Comprimento da pétala (cm)')
plt.ylabel('Frequência')
plt.show()

plt.scatterplot(data=df, x='sepal length (cm)', y='petal length (cm)', hue='species')
plt.title('Comprimento da sépala vs. comprimento da pétala')
plt.xlabel('Comprimento da sépala (cm)')
plt.ylabel('Comprimento da pétala (cm)')
plt.show()

plt.boxplot(data=df, x='species', y='petal width (cm)')
plt.title('Largura da pétala por espécie')
plt.xlabel('Espécie')
plt.ylabel('Largura da pétala (cm)')
plt.show()

plt.boxplot(data=df, x='species', y='petal width (cm)')
plt.title('Largura da pétala por espécie')
plt.xlabel('Espécie')
plt.ylabel('Largura da pétala (cm)')
plt.show()

correlacao = df.drop(columns=['target', 'species']).corr()
correlacao

## 1. O dataset está perfeitamente balanceado, com 50 amostras para cada uma das três espécies (setosa, versicolor e virginica).

## 2. A espécie *setosa* se destaca claramente das outras duas: possui pétalas muito menores (comprimento médio de 1.46 cm contra 4.26 cm e 5.55 cm das demais), o que a torna facilmente separável no gráfico de dispersão.

## 3. As variáveis "comprimento da pétala" e "largura da pétala" possuem correlação muito forte (0.96), indicando que crescem praticamente juntas. Já "largura da sépala" é a única medida que se correlaciona negativamente com as demais.

## 4. Apesar de o dataset ser tratado como "limpo" (sem valores ausentes), foi identificada 1 linha duplicada, o que reforça a importância de sempre checar duplicatas mesmo em