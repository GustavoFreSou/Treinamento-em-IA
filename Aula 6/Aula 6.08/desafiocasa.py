import re
import numpy as np

def tokenizar(texto):
    return re.findall(r"[a-záàâãéêíóôõúç]+", texto.lower())

def construir_vocabulario(textos_treino):
    palavras = set()
    for txt in textos_treino:
        palavras.update(tokenizar(txt))
    return {p: j for j, p in enumerate(sorted(palavras))}

def vetorizar(textos, vocab):
    X = np.zeros((len(textos), len(vocab)), dtype=int)
    for i, txt in enumerate(textos):
        for t in tokenizar(txt):
            if t in vocab:
                X[i, vocab[t]] += 1
    return X

def matriz_confusao(y_true, y_pred, classes):
    M = np.zeros((len(classes), len(classes)), dtype=int)
    for real, prev in zip(y_true, y_pred):
        M[np.where(classes == real)[0][0], np.where(classes == prev)[0][0]] += 1
    return M

class NaiveBayesDoZero:

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        priori = np.array([np.mean(y == c) for c in self.classes_])
        self.contagens_ = np.array([X[y == c].sum(axis=0) for c in self.classes_])
        totais = self.contagens_.sum(axis=1, keepdims=True)
        prob = (self.contagens_ + self.alpha) / (totais + self.alpha * X.shape[1])
        self.log_priori_ = np.log(priori)
        self.log_prob_ = np.log(prob)
        return self

    def log_pontuacoes(self, X):
        return X @ self.log_prob_.T + self.log_priori_

    def predict(self, X):
        S = self.log_pontuacoes(X)
        return self.classes_[np.argmax(S, axis=1)]

    def predict_proba(self, X):
        S = self.log_pontuacoes(X)
        S = S - S.max(axis=1, keepdims=True)
        E = np.exp(S)
        return E / E.sum(axis=1, keepdims=True)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)

spam = [
    "ganhe dinheiro rápido clique agora",
    "oferta imperdível desconto exclusivo hoje",
    "você foi sorteado ganhe um prêmio grátis",
    "parabéns você ganhou um celular novo",
    "clique no link e receba seu bônus grátis",
    "promoção relâmpago compre agora com desconto",
    "empréstimo aprovado sem consulta ao spc",
    "seu cartão foi bloqueado clique para desbloquear",
    "ganhe seguidores grátis agora mesmo",
    "última chance de garantir sua oferta exclusiva",
    "renda extra trabalhando em casa ganhe muito",
    "cassino online bônus de boas vindas grátis",
    "emagreça rápido com este produto milagroso",
    "liquidação total tudo com cinquenta por cento de desconto",
    "você tem um prêmio esperando clique aqui",
    "dinheiro fácil sem sair de casa",
    "urgente confirme seus dados para receber o prêmio",
    "frete grátis em todas as compras hoje",
    "ganhe um cupom de desconto exclusivo",
    "oferta limitada aproveite agora e ganhe brinde",
    "sua conta será suspensa clique agora para regularizar",
    "investimento garantido lucro rápido sem risco",
    "resgate seu bônus grátis antes que expire",
    "promoção imperdível compre um e leve dois",
]

normais = [
    "a reunião foi adiada para amanhã às dez",
    "pode me enviar o relatório até sexta",
    "vamos almoçar juntos hoje",
    "não esqueça de comprar leite no mercado",
    "a aula de amanhã começa mais cedo",
    "obrigado pela ajuda com o trabalho",
    "estou chegando em casa daqui a pouco",
    "o professor liberou as notas da prova",
    "te ligo mais tarde para conversar",
    "feliz aniversário muita saúde e alegria",
    "preciso remarcar a consulta da semana que vem",
    "o jantar de domingo na casa da vovó",
    "segue em anexo a ata da reunião",
    "você viu o jogo ontem à noite",
    "vou sair do trabalho um pouco mais tarde",
    "me avise quando chegar na escola",
    "o projeto está quase pronto para entrega",
    "vamos marcar o cinema no fim de semana",
    "lembrete da consulta com o dentista amanhã",
    "obrigado por vir à festa ontem",
    "o ônibus está atrasado hoje",
    "pode revisar o código antes da reunião",
    "mamãe pediu para ligar quando puder",
    "a prova de matemática será na próxima semana",
]

textos = spam + normais
y = np.array([1] * len(spam) + [0] * len(normais))

rng = np.random.RandomState(42)
indices = rng.permutation(len(textos))
n_teste = len(textos) // 4
teste = indices[:n_teste]
treino = indices[n_teste:]

textos_treino = [textos[i] for i in treino]
textos_teste = [textos[i] for i in teste]
y_treino = y[treino]
y_teste = y[teste]

vocab = construir_vocabulario(textos_treino)
X_treino = vetorizar(textos_treino, vocab)
X_teste = vetorizar(textos_teste, vocab)

assert set(treino).isdisjoint(set(teste))
assert X_treino.shape[1] == X_teste.shape[1] == len(vocab)

modelo = NaiveBayesDoZero(alpha=1.0)
modelo.fit(X_treino, y_treino)

assert np.allclose(np.exp(modelo.log_prob_).sum(axis=1), 1)

previsoes = modelo.predict(X_teste)
probas = modelo.predict_proba(X_teste)

assert np.allclose(probas.sum(axis=1), 1)
assert np.array_equal(previsoes, modelo.classes_[np.argmax(modelo.log_pontuacoes(X_teste), axis=1)])

print("Treino:", len(treino), "| Teste:", len(teste))
print("Valores reais:", y_teste)
print("Previsões:    ", previsoes)
print("Acurácia:", modelo.score(X_teste, y_teste))
print("Matriz de confusão (linhas = real, colunas = previsto):")
print(matriz_confusao(y_teste, previsoes, modelo.classes_))

erros = [i for i in range(len(y_teste)) if previsoes[i] != y_teste[i]]
for i in erros:
    print("Erro:", textos_teste[i], "| real:", y_teste[i], "| previsto:", previsoes[i])

mini_textos = [
    "promoção grátis grátis",
    "ganhe promoção",
    "reunião hoje",
    "agenda reunião",
]
mini_y = np.array([1, 1, 0, 0])
mini_vocab = construir_vocabulario(mini_textos)
mini_modelo = NaiveBayesDoZero(alpha=1.0).fit(vetorizar(mini_textos, mini_vocab), mini_y)
p_mini = mini_modelo.predict_proba(vetorizar(["grátis promoção"], mini_vocab))[0, 1]
print("P(spam | grátis promoção):", round(p_mini, 4))
assert abs(p_mini - 0.8815) < 1e-4

novas = [
    "Parabéns! Você ganhou um prêmio grátis, clique agora!",
    "Vamos marcar a reunião de amanhã para conversar sobre o projeto",
]

X_novas = vetorizar(novas, vocab)
classes_novas = modelo.predict(X_novas)
probas_novas = modelo.predict_proba(X_novas)

for msg, c, p in zip(novas, classes_novas, probas_novas):
    print(msg)
    print("Classe:", "Spam" if c == 1 else "Normal")
    print("Probabilidades [normal, spam]:", p)

def explicar(mensagem):
    S = modelo.log_pontuacoes(vetorizar([mensagem], vocab))[0]
    print(mensagem)
    print("Log-priori [normal, spam]:", modelo.log_priori_)
    for t in tokenizar(mensagem):
        if t in vocab:
            j = vocab[t]
            diff = modelo.log_prob_[1, j] - modelo.log_prob_[0, j]
            print(f"  {t}: log P(p|spam) - log P(p|normal) = {diff:+.3f}")
        else:
            print(f"  {t}: fora do vocabulário")
    print("Log-pontuações [normal, spam]:", S)

for msg in novas:
    explicar(msg)
