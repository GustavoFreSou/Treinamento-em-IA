
alunos = ["Evelyn","Marcelo","Pedro","Gustavo","Matheus","Felipe"]
nota = [8,6,9,5,7,10]

for aluno in alunos:
   print(aluno)


reprovados =0
aprovados = 0
analise = 0

while analise <1 :

    if (nota)[0] >= 7:
        aprovados +=1
    else:
        reprovados +=1
    if (nota)[1] >= 7:
        aprovados +=1
    else:
        reprovados +=1
    if (nota)[2] >= 7:
        aprovados +=1
    else:
        reprovados +=1
    if (nota)[3] >= 7:
        aprovados +=1
    else:
        reprovados +=1
    if (nota)[4] >= 7:
        aprovados +=1
    else:
        reprovados +=1
    if (nota)[5] >= 7:
        aprovados +=1
    else:
        reprovados +=1


    analise +=1    
print("TOTAL DE ALUNOS: 6 ", alunos)
print("APROVADOS: " , aprovados)
print("REPROVADOS: " , reprovados)
