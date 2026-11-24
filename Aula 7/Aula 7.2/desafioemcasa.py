import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

X, y = make_moons(n_samples=300, noise=0.20, random_state=42)

print(X.shape)
print(y.shape)
print(np.unique(y, return_counts=True))

plt.figure(figsize=(6, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1],
            c="#B2B9C1", marker="s",
            edgecolor="k", label="Classe 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1],
            c="#FF6B35", marker="o",
            edgecolor="k", label="Classe 1")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend(loc="upper center", ncol=2,
           bbox_to_anchor=(0.5, -0.1), frameon=False)
plt.axis("equal")
plt.show()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)
print(X_train.shape, X_test.shape)

svm_linear = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="linear", C=1.0))
])
svm_linear.fit(X_train, y_train)

svm_rbf = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC(kernel="rbf", C=1.0, gamma="scale"))
])
svm_rbf.fit(X_train, y_train)

pred_linear = svm_linear.predict(X_test)
pred_rbf = svm_rbf.predict(X_test)

print("Reais:           ", y_test[:15])
print("Previstos Linear:", pred_linear[:15])
print("Previstos RBF:   ", pred_rbf[:15])

acc_linear = accuracy_score(y_test, pred_linear)
acc_rbf = accuracy_score(y_test, pred_rbf)
print(f"Acurácia (linear): {acc_linear:.3f}")
print(f"Acurácia (RBF): {acc_rbf:.3f}")

cm_linear = confusion_matrix(y_test, pred_linear)
cm_rbf = confusion_matrix(y_test, pred_rbf)
print("Matriz de confusão (linear):")
print(cm_linear)
print("Matriz de confusão (RBF):")
print(cm_rbf)

def plot_decision_boundary(model, X, y, ax, title,
                           x_lim, y_lim, resolution=300):
    xx, yy = np.meshgrid(
        np.linspace(x_lim[0], x_lim[1], resolution),
        np.linspace(y_lim[0], y_lim[1], resolution),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(grid).reshape(xx.shape)
    D = model.decision_function(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, levels=[-0.5, 0.5, 1.5],
                colors=["#E7E5E1", "#F8DED1"])
    ax.contour(xx, yy, D, levels=[-1, 0, 1],
               colors="k", linestyles=["--", "-", "--"])
    ax.scatter(X[y == 0, 0], X[y == 0, 1],
               c="#B2B9C1", marker="s",
               edgecolor="k", s=30, label="Classe 0")
    ax.scatter(X[y == 1, 0], X[y == 1, 1],
               c="#FF6B35", marker="o",
               edgecolor="k", s=30, label="Classe 1")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal")
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_title(title)
    ax.legend(loc="upper center", ncol=3,
              frameon=False, bbox_to_anchor=(0.5, -0.12))

x_lim = (X[:, 0].min() - 0.3, X[:, 0].max() + 0.3)
y_lim = (X[:, 1].min() - 0.3, X[:, 1].max() + 0.3)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
plot_decision_boundary(svm_linear, X_train, y_train, axes[0],
                       f"SVM Linear | teste {acc_linear:.3f}", x_lim, y_lim)
plot_decision_boundary(svm_rbf, X_train, y_train, axes[1],
                       f"SVM RBF | teste {acc_rbf:.3f}", x_lim, y_lim)
plt.tight_layout()
plt.show()

valores_C = [0.01, 1, 100]
modelos_C = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, C in zip(axes, valores_C):
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=C,
                    gamma=1.0)),
    ])
    modelo.fit(X_train, y_train)
    modelos_C[C] = modelo
    acc_tr = accuracy_score(
        y_train, modelo.predict(X_train))
    acc_te = accuracy_score(
        y_test, modelo.predict(X_test))
    plot_decision_boundary(
        modelo, X_train, y_train, ax,
        f"C={C} | treino {acc_tr:.3f}"
        f" | teste {acc_te:.3f}",
        x_lim, y_lim)
plt.tight_layout()
plt.show()

valores_gamma = [0.1, 1, 10]
modelos_gamma = {}
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
for ax, g in zip(axes, valores_gamma):
    modelo = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", C=1.0,
                    gamma=g)),
    ])
    modelo.fit(X_train, y_train)
    modelos_gamma[g] = modelo
    acc_tr = accuracy_score(
        y_train, modelo.predict(X_train))
    acc_te = accuracy_score(
        y_test, modelo.predict(X_test))
    plot_decision_boundary(
        modelo, X_train, y_train, ax,
        f"gamma={g} | treino {acc_tr:.3f}"
        f" | teste {acc_te:.3f}",
        x_lim, y_lim)
plt.tight_layout()
plt.show()

def resumo(nome, modelo):
    svc = modelo.named_steps["svc"]
    g = "-" if svc.kernel == "linear" else svc.gamma
    return [nome, svc.kernel, svc.C, g,
            accuracy_score(y_train, modelo.predict(X_train)),
            accuracy_score(y_test, modelo.predict(X_test)),
            int(svc.n_support_.sum())]

linhas = [resumo("Linear", svm_linear),
          resumo("RBF scale", svm_rbf)]
linhas += [resumo(f"RBF C={c}", m)
           for c, m in modelos_C.items()]
linhas += [resumo(f"RBF gamma={g}", m)
           for g, m in modelos_gamma.items()]

print(f"{'Modelo':<15}{'Kernel':<8}{'C':>7}"
      f"{'gamma':>8}{'Treino':>9}{'Teste':>8}{'VS':>6}")
for nome, k, c, g, tr, te, nv in linhas:
    print(f"{nome:<15}{k:<8}{c:>7}{str(g):>8}"
          f"{tr:>9.3f}{te:>8.3f}{nv:>6}")
