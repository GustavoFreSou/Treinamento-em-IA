import numpy as np

y_real = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 1, 1, 1, 0, 0, 0, 1, 1])

acertos = (y_real == y_pred)
print("acertos:", acertos.sum())
print("accuracy:", acertos.mean())

def contar(y_real, y_pred):
    tp = np.sum((y_real == 1) & (y_pred == 1))
    tn = np.sum((y_real == 0) & (y_pred == 0))
    fp = np.sum((y_real == 0) & (y_pred == 1))
    fn = np.sum((y_real == 1) & (y_pred == 0))
    return tp, tn, fp, fn

tp, tn, fp, fn = contar(y_real, y_pred)
print("TP:", tp, "TN:", tn, "FP:", fp, "FN:", fn)

def matriz_confusao(y_real, y_pred):
    tp, tn, fp, fn = contar(y_real, y_pred)
    return np.array([[tn, fp],
                     [fn, tp]])

print(matriz_confusao(y_real, y_pred))

def divide(num, den):
    return num / den if den != 0 else 0.0

def precision(tp, fp):
    return divide(tp, tp + fp)

p = precision(tp, fp)
print("precision:", p)

def recall(tp, fn):
    return divide(tp, tp + fn)

r = recall(tp, fn)
print("recall:", r)

def f1_score(p, r):
    return divide(2 * p * r, p + r)

print("F1:", f1_score(p, r))

y_real3 = np.array([0]*10 + [1]*5 + [2]*10)
y_pred3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                    0, 1, 1, 1, 2,
                    1, 2, 2, 2, 2, 2, 2, 2, 2, 2])

classes = np.unique(np.concatenate([y_real3, y_pred3]))
k = len(classes)

M = np.zeros((k, k), dtype=int)
for real, prev in zip(y_real3, y_pred3):
    M[real, prev] += 1
print(M)

def razao(num, den):
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den != 0)
    return out

tp3 = np.diag(M)
fp3 = M.sum(axis=0) - tp3     
fn3 = M.sum(axis=1) - tp3      
prec3 = razao(tp3, tp3 + fp3)
rec3  = razao(tp3, tp3 + fn3)
f13   = razao(2 * prec3 * rec3, prec3 + rec3)
support = M.sum(axis=1)
acc3 = tp3.sum() / M.sum()

nomes = ["Adelie", "Chinstrap", "Gentoo"]
print(f"{'':>14}{'precision':>11}{'recall':>9}{'f1-score':>10}{'support':>9}")
for i in range(k):
    print(f"{nomes[i]:>14}{prec3[i]:>11.2f}{rec3[i]:>9.2f}{f13[i]:>10.2f}{support[i]:>9}")
print(f"{'accuracy':>14}{'':>20}{acc3:>10.2f}{support.sum():>9}")
print(f"{'macro avg':>14}{prec3.mean():>11.2f}{rec3.mean():>9.2f}{f13.mean():>10.2f}{support.sum():>9}")
print(f"{'weighted avg':>14}"
      f"{np.average(prec3, weights=support):>11.2f}"
      f"{np.average(rec3, weights=support):>9.2f}"
      f"{np.average(f13, weights=support):>10.2f}{support.sum():>9}")

scores = np.array([0.95, 0.10, 0.70, 0.55, 0.85, 0.38, 0.45, 0.25, 0.62, 0.78])

def curva_roc(y_real, scores):
    ordem = np.argsort(-scores, kind="mergesort")
    y, s = y_real[ordem], scores[ordem]
    fim = np.r_[np.where(np.diff(s) != 0)[0], len(s) - 1]
    tp = np.cumsum(y)[fim]
    fp = (fim + 1) - tp
    P, N = y.sum(), len(y) - y.sum()
    tpr = np.r_[0, tp] / P
    fpr = np.r_[0, fp] / N
    return fpr, tpr

def auc(fpr, tpr):
    larg = fpr[1:] - fpr[:-1]
    alt = (tpr[1:] + tpr[:-1]) / 2
    return np.sum(larg * alt)

fpr, tpr = curva_roc(y_real, scores)
print("FPR:", fpr)
print("TPR:", tpr)
print("AUC:", auc(fpr, tpr))
