import pandas as pd 

dados = {
    "alunos": ["Evelyn","Marcelo","Pedro","Gustavo","Matheus","Felipe",],
    "idade": [21,25,19,23,22,18],
    "nota": [70,65,71,75,55,90],
    "curso": ["GES","GET","GEC","GEA","GEB","GEP"],
}
df = pd.DataFrame(dados)

print(df.head(5))
print("\n")

print(df.tail(5))
print("\n")

print(df.shape)
print("\n")

print(df.columns)
print("\n")

print(df.info(5))
print("\n")

print(df.describe())
print("\n")

print(df["nota"])
print("\n")

print(df[["alunos","nota"]])
print("\n")

print(df[df["nota"] >= 70])
print("\n")

print(df.sort_values("nota"))
print("\n")

print(df.sort_values("nota", ascending=False))
print("\n")

print(df.isnull().sum())
print("\n")