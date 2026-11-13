from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dados = {
    "Nome": ["Evelyn", "Pedro", "Gustavo", "Mateus", "Felipe", "Marcelo"],
    "Idade": [20, 21, 19, 22, 23, 24],
    "Nota": [8, 7, 9, 6, 5, 10]
}
df = pd.DataFrame(dados)

df.head()

df.columns

df.info()

df.isnull().sum()

df.describe()

df['Nota'].value_counts().sort_index()

df['Situacao'] = df['Nota'].apply(lambda x: 'Aprovado' if x >= 6 else 'Reprovado')
df.groupby('Situacao')[['Idade', 'Nota']].mean()


plt.bar(df['Nome'], df['Nota'], color='#4C72B0')
plt.xlabel('Aluno')
plt.ylabel('Nota')
plt.title('Notas dos alunos')
plt.show()

df['Nota'].plot(kind='hist', bins=5, color='teal', edgecolor='black')
plt.xlabel('Nota')
plt.ylabel('Frequência')
plt.title('Distribuição das notas')
plt.show()

cores = df['Situacao'].map({'Aprovado': 'green', 'Reprovado': 'red'})
plt.scatter(df['Idade'], df['Nota'], c=cores)
plt.xlabel('Idade')
plt.ylabel('Nota')
plt.title('Idade x Nota')
plt.show()

plt.boxplot(df['Nota'])
plt.title('Distribuição das notas')
plt.ylabel('Nota')
plt.show()

df[['Idade', 'Nota']].corr()

##1. A turma tem 6 alunos, com idades entre 19 e 24 anos e notas entre 5 e 10.

##2. A maioria dos alunos foi aprovada (nota ≥ 6): apenas o Felipe, com nota 5, ficou reprovado.

##3. Não há correlação forte entre idade e nota — alunos mais velhos não necessariamente tiraram notas melhores (ex: Marcelo, 24 anos, tirou 10, mas Gustavo, 19 anos, também teve uma das melhores notas, 9).

##4. Não há valores ausentes nem duplicatas no dataset, então não foi necessário nenhum tratamento de limpeza.
   