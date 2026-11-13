import matplotlib.pyplot as plt

Alunos = ["Evelyn","Marcelo","Pedro","Gustavo","Mateus","Felipe"]

Periodo = ["P3","P4","P5","P6","P7","P8"]


plt.bar(Alunos, Periodo)

plt.xlabel("Alunos")
plt.ylabel("Periodos")
plt.title("Periodos dos Alunos")

plt.show()

plt.hist(Periodo)

plt.xlabel("Alunos")
plt.ylabel("Periodos")
plt.title("Periodos dos Alunos")

plt.show()

aulas = [1,2,3,4,5,6,7,8,9,10,11,12]
faltas = [2,3,4,5,7,8,9,12]

plt.scatter(aulas,faltas)

plt.xlabel("Aulas")
plt.ylabel("Faltas")
plt.title("Frequencia")

plt.show()


