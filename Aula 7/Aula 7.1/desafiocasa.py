import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score

SEED = 42
RHO = 0.5
SCORING = "neg_root_mean_squared_error"
N_SPLITS = 5
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

print("sklearn", sklearn.__version__)

X, y = load_diabetes(return_X_y=True, as_frame=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print("treino:", X_train.shape, "| teste:", X_test.shape)

def make_pipe(model):
    return Pipeline([("scaler", StandardScaler()), ("model", model)])

MODELOS = {
    "Ridge":      lambda a: Ridge(alpha=a),
    "Lasso":      lambda a: Lasso(alpha=a, max_iter=50_000),
    "ElasticNet": lambda a: ElasticNet(alpha=a, l1_ratio=RHO, max_iter=50_000),
}
GRADE = {
    "Ridge":      np.logspace(-3, 3, 61),
    "Lasso":      np.logspace(-3, 1.5, 46),
    "ElasticNet": np.logspace(-3, 1.5, 46),
}

cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

base = cross_val_score(make_pipe(LinearRegression()), X_train, y_train, cv=cv, scoring=SCORING)
print(f"MQO (baseline) RMSE-CV = {-base.mean():.2f} ± {base.std():.2f}")


def curva_cv(nome, cv_obj):
    linhas = []
    for a in GRADE[nome]:
        s = -cross_val_score(make_pipe(MODELOS[nome](a)), X_train, y_train, cv=cv_obj, scoring=SCORING)
        linhas.append((a, s.mean(), s.std()))
    return pd.DataFrame(linhas, columns=["alpha", "rmse_medio", "rmse_dp"])

curvas = {nome: curva_cv(nome, cv) for nome in MODELOS}

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (nome, df) in zip(axes, curvas.items()):
    ax.semilogx(df.alpha, df.rmse_medio, color="C0", label="RMSE-CV (média das 5 dobras)")
    ax.fill_between(df.alpha, df.rmse_medio - df.rmse_dp, df.rmse_medio + df.rmse_dp,
                    color="C0", alpha=0.2, label="± 1 dp entre dobras")
    i = df.rmse_medio.idxmin()
    ax.axvline(df.alpha[i], color="C3", ls="--", label=f"mínimo: α = {df.alpha[i]:.3g}")
    ax.axhline(-base.mean(), color="gray", ls=":", label=f"MQO = {-base.mean():.1f}")
    titulo = nome + (f" (ρ = {RHO})" if nome == "ElasticNet" else "")
    ax.set_title(titulo)
    ax.set_xlabel("α (escala log)")
    ax.legend(fontsize=8)
axes[0].set_ylabel("RMSE-CV (unidades do alvo)")
fig.suptitle("RMSE-CV × α — diabetes, treino com 5-fold", y=1.02)
plt.tight_layout()
plt.show()

minimos = pd.DataFrame({
    n: {"alpha_min": d.alpha[d.rmse_medio.idxmin()],
        "rmse_min": d.rmse_medio.min(),
        "dp_no_min": d.rmse_dp[d.rmse_medio.idxmin()]}
    for n, d in curvas.items()
}).T
print(minimos.round(3))


ALPHAS_5 = [0.01, 0.1, 1, 10, 30]
scaler_tr = StandardScaler().fit(X_train)
Xs_tr = scaler_tr.transform(X_train)

def coefs(nome, a):
    return MODELOS[nome](a).fit(Xs_tr, y_train).coef_

tab_zeros = pd.DataFrame(
    {nome: [int((np.abs(coefs(nome, a)) < 1e-8).sum()) for a in ALPHAS_5] for nome in MODELOS},
    index=pd.Index(ALPHAS_5, name="alpha"),
)
print("Nº de coeficientes zerados (de 10):")
print(tab_zeros)

zeros_lasso = pd.DataFrame(
    {f"α={a:g}": pd.Series(coefs("Lasso", a) == 0, index=X.columns).map({True: "zero", False: "—"})
     for a in ALPHAS_5}
)
print(zeros_lasso)


def regra_1dp(df):
    i = df.rmse_medio.idxmin()
    limite = df.rmse_medio[i] + df.rmse_dp[i]
    alpha_1dp = df.alpha[df.rmse_medio <= limite].max()
    return df.alpha[i], df.rmse_medio[i], limite, alpha_1dp, df.rmse_medio[df.alpha == alpha_1dp].iloc[0]

linhas = []
for nome, df in curvas.items():
    a_min, r_min, lim, a_1dp, r_1dp = regra_1dp(df)
    n_zero = int((np.abs(coefs(nome, a_1dp)) < 1e-8).sum())
    linhas.append([nome, a_min, r_min, lim, a_1dp, r_1dp, n_zero])

regra = pd.DataFrame(linhas, columns=["modelo", "alpha_min", "rmse_min", "limite_1dp",
                                      "alpha_1dp", "rmse_no_alpha_1dp", "zeros_no_alpha_1dp"]).set_index("modelo")
print(regra.round(3))

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (nome, df) in zip(axes, curvas.items()):
    a_min, r_min, lim, a_1dp, r_1dp = regra_1dp(df)
    ax.semilogx(df.alpha, df.rmse_medio, color="C0", label="RMSE-CV")
    ax.axhline(lim, color="gray", ls=":", label=f"mínimo + 1 dp = {lim:.2f}")
    ax.axvline(a_min, color="C3", ls="--", label=f"α mínimo = {a_min:.3g}")
    ax.axvline(a_1dp, color="C2", ls="--", label=f"α 1-dp = {a_1dp:.3g}")
    ax.set_title(nome); ax.set_xlabel("α (escala log)"); ax.legend(fontsize=8)
axes[0].set_ylabel("RMSE-CV (unidades do alvo)")
plt.tight_layout(); plt.show()


SEEDS = [0, 42, 123]
res_seed, zeros_seed = {}, {}

for sd in SEEDS:
    cv_s = KFold(n_splits=N_SPLITS, shuffle=True, random_state=sd)
    df = curva_cv("Lasso", cv_s)
    a_min, r_min, lim, a_1dp, _ = regra_1dp(df)
    res_seed[sd] = {"alpha_min": a_min, "alpha_1dp": a_1dp}
    zeros_seed[f"seed {sd}"] = pd.Series(coefs("Lasso", a_1dp) == 0, index=X.columns)

print("α escolhido por semente:")
print(pd.DataFrame(res_seed).T.round(4))

zs = pd.DataFrame(zeros_seed)
zs["vezes_zerada"] = zs.sum(axis=1)
print(zs.astype({c: int for c in zs.columns}).sort_values("vezes_zerada", ascending=False))

estaveis = zs.index[zs["vezes_zerada"] == len(SEEDS)].tolist()
instaveis = zs.index[(zs["vezes_zerada"] > 0) & (zs["vezes_zerada"] < len(SEEDS))].tolist()
print("Zeradas nas 3 sementes (estáveis):", estaveis)
print("Zeradas só em algumas (instáveis):", instaveis)


resumo = regra[["alpha_min", "rmse_min", "alpha_1dp", "rmse_no_alpha_1dp", "zeros_no_alpha_1dp"]].copy()
resumo.loc["MQO"] = [np.nan, -base.mean(), np.nan, np.nan, 0]
resumo = resumo.sort_values("rmse_min")
print(resumo.round(3))


melhor_rmse = regra["rmse_min"].min()
dp_melhor = curvas[regra["rmse_min"].idxmin()].rmse_dp.iloc[curvas[regra["rmse_min"].idxmin()].rmse_medio.idxmin()]
elegiveis = regra[regra["rmse_no_alpha_1dp"] <= melhor_rmse + dp_melhor]
escolhido = elegiveis["zeros_no_alpha_1dp"].idxmax()
alpha_final = regra.loc[escolhido, "alpha_1dp"]
print(f"Escolhido: {escolhido} com α = {alpha_final:.4g}")

modelo_final = make_pipe(MODELOS[escolhido](alpha_final)).fit(X_train, y_train)
y_pred = modelo_final.predict(X_test)
rmse_te = mean_squared_error(y_test, y_pred) ** 0.5
r2_te = r2_score(y_test, y_pred)
print(f"{escolhido}: RMSE teste = {rmse_te:.2f} | R² = {r2_te:.3f}")

coef_final = pd.Series(modelo_final.named_steps["model"].coef_, index=X.columns).round(1)
print(coef_final.to_frame("coeficiente (padronizado)"))