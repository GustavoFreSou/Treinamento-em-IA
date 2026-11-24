import numpy as np
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=True)

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

Y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

print("ENTRADAS:")
print(X)
print("SAIDAS XOR:")
print(Y.ravel())

plt.figure(figsize=(6, 5))
plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(),
            cmap="coolwarm", s=300, edgecolors="black")

for (x1, x2), y in zip(X, Y.ravel()):
    plt.annotate(str(int(y)), (x1, x2),
                 xytext=(0, 10), textcoords="offset points",
                 ha="center")

plt.title("Distribuicao XOR")
plt.xlabel("x1")
plt.ylabel("x2")
plt.xticks([0, 1])
plt.yticks([0, 1])
plt.grid(alpha=0.25)
plt.show()

def treinar_perceptron(X, y, taxa_aprendizado=0.1, max_epocas=100):
    pesos = np.zeros(X.shape[1])
    bias = 0.0

    for epoca in range(max_epocas):
        erros = 0

        for xi, yi in zip(X, y):
            z = np.dot(xi, pesos) + bias
            previsao = 1 if z >= 0 else -1

            if previsao != yi:
                pesos += taxa_aprendizado * yi * xi
                bias += taxa_aprendizado * yi
                erros += 1

        if erros == 0:
            break

    return pesos, bias, epoca + 1

Y_and = np.array([-1, -1, -1, 1])

pesos_and, bias_and, epocas_and = treinar_perceptron(X, Y_and)

previsoes_and = np.where(X @ pesos_and + bias_and >= 0, 1, -1)

print("\nPERCEPTRON AND")
print("Pesos:", pesos_and)
print("Bias:", bias_and)
print("Epocas:", epocas_and)
print("Previsoes:", previsoes_and)
print("Esperado:", Y_and)
print("Acertos:", np.sum(previsoes_and == Y_and), "de 4")

Y_xor_pm = np.array([-1, 1, 1, -1])

pesos_xor, bias_xor, epocas_xor = treinar_perceptron(X, Y_xor_pm)

previsoes_xor = np.where(X @ pesos_xor + bias_xor >= 0, 1, -1)

print("\nPERCEPTRON XOR")
print("Pesos:", pesos_xor)
print("Bias:", bias_xor)
print("Epocas:", epocas_xor)
print("Previsoes:", previsoes_xor)
print("Esperado:", Y_xor_pm)
print("Acertos:", np.sum(previsoes_xor == Y_xor_pm), "de 4")

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

rng = np.random.default_rng(42)

n_entradas = 2
n_ocultos = 4
n_saidas = 1

params = {
    "W1": rng.normal(0, 0.8, (n_entradas, n_ocultos)),
    "b1": np.zeros((1, n_ocultos)),
    "W2": rng.normal(0, 0.8, (n_ocultos, n_saidas)),
    "b2": np.zeros((1, n_saidas))
}

print("\nFORMATOS DOS PARAMETROS")

for nome, valor in params.items():
    print(nome, valor.shape)

def forward(X, params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = sigmoid(Z1)

    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = sigmoid(Z2)

    cache = {
        "X": X,
        "A1": A1,
        "A2": A2
    }

    return A2, cache

def calcular_loss(Y, probabilidades):
    epsilon = 1e-12
    probabilidades = np.clip(probabilidades, epsilon, 1 - epsilon)

    loss = -np.mean(
        Y * np.log(probabilidades)
        + (1 - Y) * np.log(1 - probabilidades)
    )

    return float(loss)

A2_inicial, cache_inicial = forward(X, params)

loss_inicial = calcular_loss(Y, A2_inicial)

print("\nFORWARD PROPAGATION")
print("Previsoes iniciais:", A2_inicial.ravel())
print("Formato A1:", cache_inicial["A1"].shape)
print("Formato A2:", A2_inicial.shape)
print("Perda inicial:", loss_inicial)

def backward(Y, params, cache):
    m = Y.shape[0]

    X = cache["X"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = (A2 - Y) / m

    dW2 = A1.T @ dZ2
    db2 = np.sum(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ params["W2"].T
    dZ1 = dA1 * A1 * (1 - A1)

    dW1 = X.T @ dZ1
    db1 = np.sum(dZ1, axis=0, keepdims=True)

    return {
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2
    }

def update_parameters(params, gradients, learning_rate):
    for key in params:
        params[key] -= learning_rate * gradients[key]

    return params

learning_rate = 0.5
epochs = 20000
historico_loss = []

for epoca in range(epochs):
    A2, cache = forward(X, params)

    loss = calcular_loss(Y, A2)

    gradients = backward(Y, params, cache)

    params = update_parameters(params, gradients, learning_rate)

    if epoca % 100 == 0:
        historico_loss.append(loss)

probabilidades_finais, _ = forward(X, params)
loss_final = calcular_loss(Y, probabilidades_finais)

print("\nRESULTADO DO TREINAMENTO")
print(f"Perda inicial: {loss_inicial:.6f}")
print(f"Perda final: {loss_final:.6f}")

plt.figure(figsize=(8, 4.5))
plt.plot(np.arange(len(historico_loss)) * 100,
         historico_loss, linewidth=2)

plt.xlabel("Epocas")
plt.ylabel("Binary Cross-Entropy")
plt.title("Curva de aprendizado da MLP")
plt.grid(alpha=0.25)
plt.show()

previsoes_binarias = (probabilidades_finais >= 0.5).astype(int)

print("\nVERIFICACAO DAS PREVISOES")
print("Entrada | Probabilidade | Previsto | Esperado")

for entrada, prob, previsto, esperado in zip(
    X,
    probabilidades_finais.ravel(),
    previsoes_binarias.ravel(),
    Y.ravel()
):
    print(
        f"{entrada.astype(int)} | "
        f"{prob:.6f} | "
        f"{previsto} | "
        f"{int(esperado)}"
    )

print("\nPrevisoes:", previsoes_binarias.ravel())
print("Esperado: ", Y.ravel().astype(int))
print("Acertos:", np.sum(previsoes_binarias == Y), "de 4")

x1 = np.linspace(-0.5, 1.5, 300)
x2 = np.linspace(-0.5, 1.5, 300)

G1, G2 = np.meshgrid(x1, x2)

grade = np.column_stack([G1.ravel(), G2.ravel()])

saida_grade, _ = forward(grade, params)
saida_grade = saida_grade.reshape(G1.shape)

plt.figure(figsize=(7, 5.5))

plt.contourf(G1, G2, saida_grade,
             levels=20, cmap="coolwarm", alpha=0.85)

plt.contour(G1, G2, saida_grade,
            levels=[0.5], colors="black", linewidths=2)

plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(),
            cmap="coolwarm", s=220, edgecolors="black")

for (x1_pt, x2_pt), y in zip(X, Y.ravel()):
    plt.annotate(str(int(y)), (x1_pt, x2_pt),
                 xytext=(0, 9), textcoords="offset points",
                 ha="center")

plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Regiao de decisao da MLP")
plt.grid(alpha=0.15)
plt.show()


print("\nVARIANDO NEURONIOS OCULTOS E TAXA DE APRENDIZADO")

def testar_configuracao(n_ocultos, learning_rate, epochs=20000):
    rng = np.random.default_rng(42)

    parametros = {
        "W1": rng.normal(0, 0.8, (2, n_ocultos)),
        "b1": np.zeros((1, n_ocultos)),
        "W2": rng.normal(0, 0.8, (n_ocultos, 1)),
        "b2": np.zeros((1, 1))
    }

    for _ in range(epochs):
        A2, cache = forward(X, parametros)
        gradients = backward(Y, parametros, cache)
        parametros = update_parameters(
            parametros, gradients, learning_rate
        )

    probabilidades, _ = forward(X, parametros)
    perda = calcular_loss(Y, probabilidades)
    previsoes = (probabilidades >= 0.5).astype(int)

    acertos = np.sum(previsoes == Y)

    return perda, acertos

for neuronios in [2, 4, 8, 16]:
    for taxa in [0.1, 0.5, 1.0]:
        perda, acertos = testar_configuracao(neuronios, taxa)

        print(
            f"Neuronios: {neuronios:2d} | "
            f"Taxa: {taxa:.1f} | "
            f"Perda: {perda:.6f} | "
            f"Acertos: {acertos}/4"
        )

resultado_esperado = np.array([0, 1, 1, 0])
resultado_obtido = previsoes_binarias.ravel()

assert np.array_equal(resultado_obtido, resultado_esperado), (
    f"XOR nao aprendido. Obtido: {resultado_obtido}"
)

assert loss_final < loss_inicial, (
    "A perda final deveria ser menor que a inicial."
)

print("\nVERIFICACAO APROVADA!")
print("A MLP aprendeu as quatro combinacoes do XOR.")
print(f"Perda: {loss_inicial:.6f} -> {loss_final:.6f}")
