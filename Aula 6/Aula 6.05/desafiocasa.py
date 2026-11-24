import numpy as np

def razao(num, den):
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    out = np.zeros_like(num)
    np.divide(num, den, out=out, where=den != 0)
    return out

def meu_classification_report(y_real, y_pred):
    y_real, y_pred = np.asarray(y_real), np.asarray(y_pred)
    classes = np.unique(np.concatenate([y_real, y_pred]))
    k = len(classes)
    indice = {c: i for i, c in enumerate(classes)}

    M = np.zeros((k, k), dtype=int)
    for real, prev in zip(y_real, y_pred):
        M[indice[real], indice[prev]] += 1

    tp = np.diag(M)
    fp = M.sum(axis=0) - tp
    fn = M.sum(axis=1) - tp
    prec = razao(tp, tp + fp)
    rec = razao(tp, tp + fn)
    f1 = razao(2 * prec * rec, prec + rec)
    support = M.sum(axis=1)
    acc = tp.sum() / M.sum()
    total = support.sum()

    print(M)
    print(f"{'':>14}{'precision':>11}{'recall':>9}{'f1-score':>10}{'support':>9}")
    for i, c in enumerate(classes):
        print(f"{str(c):>14}{prec[i]:>11.2f}{rec[i]:>9.2f}{f1[i]:>10.2f}{support[i]:>9}")
    print(f"{'accuracy':>14}{'':>20}{acc:>10.2f}{total:>9}")
    print(f"{'macro avg':>14}{prec.mean():>11.2f}{rec.mean():>9.2f}{f1.mean():>10.2f}{total:>9}")
    print(f"{'weighted avg':>14}"
          f"{np.average(prec, weights=support):>11.2f}"
          f"{np.average(rec, weights=support):>9.2f}"
          f"{np.average(f1, weights=support):>10.2f}{total:>9}")

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

y_real = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0])
y_pred = np.array([1, 0, 1, 1, 1, 0, 0, 0, 1, 1])
scores = np.array([0.95, 0.10, 0.70, 0.55, 0.85, 0.38, 0.45, 0.25, 0.62, 0.78])

y_real3 = np.array([0] * 10 + [1] * 5 + [2] * 10)
y_pred3 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
                    0, 1, 1, 1, 2,
                    1, 2, 2, 2, 2, 2, 2, 2, 2, 2])

meu_classification_report(y_real, y_pred)
meu_classification_report(y_real3, y_pred3)

fpr, tpr = curva_roc(y_real, scores)
print("FPR:", fpr)
print("TPR:", tpr)
print("AUC:", auc(fpr, tpr))
