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

print("Formato de X:", X.shape)
print("Formato de Y:", Y.shape)
print("Entradas:\n", X)
print("Saídas XOR:\n", Y.ravel())

plt.figure(figsize=(6, 5))
plt.scatter(
    X[:, 0], X[:, 1],
    c=Y.ravel(), cmap="coolwarm", s=300,
    edgecolors="black", linewidths=1.2
)

for (x1, x2), y in zip(X, Y.ravel()):
    plt.annotate(f"{int(y)}", (x1, x2), xytext=(0, 10),
                 textcoords="offset points", ha="center", fontsize=12)

plt.xticks([0, 1])
plt.yticks([0, 1])
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("XOR: classes em diagonais opostas")
plt.grid(alpha=0.25)
plt.show()

Y_or = np.array([-1, 1, 1, 1])

print("Rótulos OR em -1/+1:", Y_or)

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

pesos_or, bias_or, epocas_or = treinar_perceptron(X, Y_or)

z_or = X @ pesos_or + bias_or
previsoes_or = np.where(z_or >= 0, 1, -1)

print("Pesos:", pesos_or)
print("Bias:", bias_or)
print("Épocas:", epocas_or)
print("Previsões:", previsoes_or)
print("Esperado: ", Y_or)
print("Acertos:", np.sum(previsoes_or == Y_or), "de", len(Y_or))

Y_xor_pm = np.array([-1, 1, 1, -1])

pesos_xor, bias_xor, epocas_xor = treinar_perceptron(
    X, Y_xor_pm, taxa_aprendizado=0.1, max_epocas=100
)

z_xor = X @ pesos_xor + bias_xor
previsoes_xor = np.where(z_xor >= 0, 1, -1)

print("Épocas executadas:", epocas_xor)
print("Pesos:", pesos_xor)
print("Bias:", bias_xor)
print("Previsões:", previsoes_xor)
print("Esperado:  ", Y_xor_pm)
print("Acertos:", np.sum(previsoes_xor == Y_xor_pm), "de", len(Y_xor_pm))

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

print("Sigmoid(-2), sigmoid(0), sigmoid(2):", sigmoid(np.array([-2.0, 0.0, 2.0])))

rng = np.random.default_rng(42)

n_entradas = 2
n_ocultos = 4
n_saidas = 1

params = {
    "W1": rng.normal(0, 0.8, size=(n_entradas, n_ocultos)),
    "b1": np.zeros((1, n_ocultos)),
    "W2": rng.normal(0, 0.8, size=(n_ocultos, n_saidas)),
    "b2": np.zeros((1, n_saidas))
}

for nome, valor in params.items():
    print(nome, valor.shape)

def forward(X, params):
    Z1 = X @ params["W1"] + params["b1"]
    A1 = sigmoid(Z1)

    Z2 = A1 @ params["W2"] + params["b2"]
    A2 = sigmoid(Z2)

    cache = {
        "X": X,
        "Z1": Z1,
        "A1": A1,
        "Z2": Z2,
        "A2": A2
    }

    return A2, cache

A2_inicial, cache_inicial = forward(X, params)

print("Previsões iniciais A2:", A2_inicial.ravel())
print("Formato A1:", cache_inicial["A1"].shape)
print("Formato A2:", A2_inicial.shape)

def calcular_loss(Y, probabilidades):
    epsilon = 1e-12
    probabilidades = np.clip(probabilidades, epsilon, 1 - epsilon)

    loss = -np.mean(
        Y * np.log(probabilidades)
        + (1 - Y) * np.log(1 - probabilidades)
    )

    return float(loss)

loss_inicial = calcular_loss(Y, A2_inicial)

print("Perda inicial:", loss_inicial)

A2_exemplo = 0.8
Y_exemplo = 1.0
A1_exemplo = 0.6
W2_exemplo = 2.0

dZ2_exemplo = A2_exemplo - Y_exemplo
dA1_exemplo = dZ2_exemplo * W2_exemplo
dZ1_exemplo = dA1_exemplo * A1_exemplo * (1 - A1_exemplo)

print("dZ2 =", dZ2_exemplo)
print("dA1 =", dA1_exemplo)
print("dZ1 =", dZ1_exemplo)

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

gradientes_iniciais = backward(Y, params, cache_inicial)

for nome, gradiente in gradientes_iniciais.items():
    print(nome, "gradiente:", gradiente.shape, "parâmetro:", params[nome].shape)

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

print(f"Perda inicial: {loss_inicial:.6f}")
print(f"Perda final:   {loss_final:.6f}")

epocas_registradas = np.arange(len(historico_loss)) * 100

plt.figure(figsize=(8, 4.5))
plt.plot(epocas_registradas, historico_loss, color="#62CDB5", linewidth=2)
plt.xlabel("Época")
plt.ylabel("Binary Cross-Entropy")
plt.title("Aprendizado da MLP para o XOR")
plt.grid(alpha=0.25)
plt.show()

probabilidades_finais, _ = forward(X, params)
previsoes_binarias = (probabilidades_finais >= 0.5).astype(int)

print("Entrada | Probabilidade | Previsto | Esperado")

for entrada, prob, previsto, esperado in zip(
    X, probabilidades_finais.ravel(),
    previsoes_binarias.ravel(), Y.ravel()
):
    print(f"{entrada.astype(int)} | {prob:.6f} | {previsto} | {int(esperado)}")

print("\nPrevisões binárias:", previsoes_binarias.ravel())
print("Rótulos esperados: ", Y.astype(int).ravel())
print("Acertos:", np.sum(previsoes_binarias == Y), "de", len(Y))

x1 = np.linspace(-0.5, 1.5, 300)
x2 = np.linspace(-0.5, 1.5, 300)

G1, G2 = np.meshgrid(x1, x2)

grade = np.column_stack([G1.ravel(), G2.ravel()])

saida_grade, _ = forward(grade, params)
saida_grade = saida_grade.reshape(G1.shape)

plt.figure(figsize=(7, 5.5))

plt.contourf(
    G1, G2, saida_grade,
    levels=20, cmap="coolwarm", alpha=0.85
)

plt.contour(
    G1, G2, saida_grade,
    levels=[0.5], colors="black", linewidths=2
)

plt.scatter(
    X[:, 0], X[:, 1], c=Y.ravel(), cmap="coolwarm",
    s=220, edgecolors="black", linewidths=1.2
)

for (x1_pt, x2_pt), y in zip(X, Y.ravel()):
    plt.annotate(
        str(int(y)), (x1_pt, x2_pt), xytext=(0, 9),
        textcoords="offset points", ha="center", fontsize=11
    )

plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Região de decisão da MLP treinada para XOR")
plt.grid(alpha=0.15)
plt.show()

resultado_esperado = np.array([0, 1, 1, 0])
resultado_obtido = previsoes_binarias.ravel()

assert np.array_equal(resultado_obtido, resultado_esperado), (
    f"A MLP não reproduziu XOR. Obtido: {resultado_obtido}; esperado: {resultado_esperado}"
)

assert loss_final < loss_inicial, "A perda final deveria ser menor que a inicial."

print("Verificação aprovada: a MLP aprendeu as quatro combinações do XOR.")
print(f"Perda: {loss_inicial:.6f} → {loss_final:.6f}")
