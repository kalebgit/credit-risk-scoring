"""
Implementacion manual de regresion logistica (Newton-Raphson / IRLS),
inferencia asintotica (Wald, LRT) y test de Box-Tidwell (univariado y
multivariado), sin usar statsmodels para el ajuste.

Objetivo: aprender el mecanismo derivado en las notas de estudio
("Regresion Logistica: MLE, Inferencia Asintotica y Diagnostico de
Linealidad en el Logit"), seccion por seccion:

  seccion 3.1-3.4 -> fit_logit_irls()      (log-verosimilitud, score,
                                             Hessiano/info de Fisher, IRLS)
  seccion 3.5 / 4.1 -> wald_scalar()       (SE = sqrt(diag(I^-1)), test z)
  seccion 4.3       -> lrt()               (razon de verosimilitudes)
  seccion 6.4 (A)   -> box_tidwell_manual()  Wald escalar por termino x*ln(x)
  seccion 6.4 (B)   -> wald_multivariado()   forma cuadratica, ecuacion (7)

Al final se compara contra sm.Logit(...).fit() con datos sinteticos donde
se conoce la relacion verdadera, para validar que los numeros coinciden.
"""

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------------
# 1. MLE por Newton-Raphson / IRLS  (PDF secciones 3.1-3.4)
# ---------------------------------------------------------------------------


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def fit_logit_irls(X, y, tol=1e-8, max_iter=100, verbose=False):
    """
    Ajusta un modelo logistico por Newton-Raphson / IRLS (ecuacion 5 del PDF).

    X: matriz (n, p) SIN columna de 1's (el intercepto se agrega aqui).
    y: vector (n,) de 0/1.

    Como el enlace logit es canonico, el Hessiano observado coincide con
    la informacion de Fisher esperada (seccion 3.3), asi que Newton-Raphson
    y Fisher scoring son el mismo algoritmo: por eso basta una funcion.
    """
    n, p = X.shape
    Xc = np.column_stack([np.ones(n), X])  # intercepto -> (n, p+1)
    beta = np.zeros(Xc.shape[1])

    for it in range(max_iter):
        eta = Xc @ beta
        pi = sigmoid(eta)

        # log-verosimilitud (ecuacion 1)
        ll = np.sum(y * eta - np.log1p(np.exp(eta)))

        # score: gradiente de ll (ecuacion 2)
        u = Xc.T @ (y - pi)

        # Hessiano / informacion de Fisher: I = X^T W X, W = diag(pi(1-pi))
        # (ecuaciones 3 y 4). No se construye W como matriz n x n densa.
        W = pi * (1 - pi)
        I_fisher = Xc.T @ (Xc * W[:, None])

        # paso de Newton-Raphson: beta_new = beta + I^-1 u  (ecuacion 5)
        delta = np.linalg.solve(I_fisher, u)
        beta = beta + delta

        if verbose:
            print(f"  iter {it:2d}  ll={ll:12.4f}  |delta|={np.linalg.norm(delta):.2e}")
        if np.linalg.norm(delta) < tol:
            break

    # recalcular en el optimo -> covarianza asintotica final (Teorema 3.2)
    eta = Xc @ beta
    pi = sigmoid(eta)
    W = pi * (1 - pi)
    I_fisher = Xc.T @ (Xc * W[:, None])
    cov = np.linalg.inv(I_fisher)  # I^-1(beta_hat)
    se = np.sqrt(np.diag(cov))
    ll_final = np.sum(y * eta - np.log1p(np.exp(eta)))

    return {"beta": beta, "se": se, "cov": cov, "ll": ll_final, "n_iter": it + 1}


def wald_scalar(beta, se, j, beta0=0.0):
    """Wald para un solo coeficiente (seccion 4.1): z = (beta_j - beta0)/SE."""
    z = (beta[j] - beta0) / se[j]
    p = 2 * stats.norm.sf(abs(z))
    return z, p


def lrt(ll_full, ll_reduced, df_diff):
    """Razon de verosimilitudes (seccion 4.3, Teorema de Wilks)."""
    g2 = 2 * (ll_full - ll_reduced)
    p = stats.chi2.sf(g2, df=df_diff)
    return g2, p


def wald_multivariado(beta, cov, idx_restringidos):
    """
    Wald conjunto H0: beta[idx] = 0 simultaneamente (seccion 6.4-B, ec. 7):
    W = gamma_hat^T [Var(gamma_hat)]^-1 gamma_hat  ~ chi2_m
    """
    idx = np.array(idx_restringidos)
    g = beta[idx]
    cov_block = cov[np.ix_(idx, idx)]
    W = g @ np.linalg.solve(cov_block, g)
    p = stats.chi2.sf(W, df=len(idx))
    return W, p


# ---------------------------------------------------------------------------
# 2. Box-Tidwell manual: univariado + multivariado (seccion 6)
# ---------------------------------------------------------------------------


def box_tidwell_manual(df, y, cols_testear, cols_control):
    """
    Igual que box_tidwell() en diagnostics2.py pero ajustando con
    fit_logit_irls() (a mano) en vez de sm.Logit, y agregando el Wald
    multivariado explicito (ecuacion 7) ademas de la LRT conjunta.
    """
    X_cols = list(cols_control)
    inter_cols = []
    df = df.copy()

    for c in cols_testear:
        v = df[c].astype(float)
        if not (v > 0).all():
            print(f"  [omitida] {c}: tiene valores <= 0")
            continue
        X_cols.append(c)
        df[f"{c}_ln"] = v * np.log(v)
        inter_cols.append(f"{c}_ln")

    all_cols = X_cols + inter_cols
    data = df[all_cols].copy()
    data["_y"] = y.values
    data = data.dropna()
    y_clean = data.pop("_y").values

    # estandarizar (mismo motivo que en el script original: evitar mala condicion)
    Xstd = (data - data.mean()) / data.std().replace(0, 1)
    X = Xstd.values

    modelo_full = fit_logit_irls(X, y_clean)
    idx_inter = [1 + all_cols.index(c) for c in inter_cols]  # +1 por intercepto

    print("== Wald escalar por termino (seccion 6.4-A) ==")
    resultados = {}
    for j, col in zip(idx_inter, inter_cols):
        var = col.replace("_ln", "")
        z, p = wald_scalar(modelo_full["beta"], modelo_full["se"], j)
        flag = "*** VIOLACION" if p < 0.05 else "OK"
        print(f"  {var:25s} z={z:7.3f}  p={p:.4f}  {flag}")
        resultados[var] = {"z": z, "p": p}

    # Wald multivariado conjunto (ecuacion 7) — H0: todos los gamma = 0 a la vez
    W, p_wald = wald_multivariado(modelo_full["beta"], modelo_full["cov"], idx_inter)
    print(
        f"\nWald multivariado conjunto: W={W:.3f}  gl={len(idx_inter)}  p={p_wald:.4f}"
    )

    # LRT conjunto (misma H0, via modelo reducido) — para comparar ambas rutas
    idx_reducido = [i for i in range(X.shape[1]) if i not in [c - 1 for c in idx_inter]]
    modelo_red = fit_logit_irls(X[:, idx_reducido], y_clean)
    g2, p_lrt = lrt(modelo_full["ll"], modelo_red["ll"], df_diff=len(idx_inter))
    print(
        f"LRT conjunto:               G2={g2:.3f}  gl={len(idx_inter)}  p={p_lrt:.4f}"
    )
    print("(Wald y LRT deben coincidir aprox. — seccion 4, Observacion 4.4)")

    return modelo_full, resultados, {"wald": (W, p_wald), "lrt": (g2, p_lrt)}


# ---------------------------------------------------------------------------
# 3. Validacion contra statsmodels con datos sinteticos
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import statsmodels.api as sm

    rng = np.random.default_rng(42)
    n = 5000
    x1 = rng.normal(size=n)
    x2 = rng.exponential(size=n) + 0.1  # positiva -> se puede testear box-tidwell
    x3 = rng.normal(size=n)

    # relacion verdadera: x2 entra como log(x2), NO lineal -> box-tidwell debe detectarlo
    logit_true = -0.5 + 0.8 * x1 + 0.6 * np.log(x2) - 0.4 * x3
    p_true = sigmoid(logit_true)
    y = rng.binomial(1, p_true)
    X = np.column_stack([x1, x2, x3])

    print("=== 1. IRLS manual vs statsmodels (mismo modelo, sin transformar x2) ===")
    modelo = fit_logit_irls(X, y, verbose=True)
    print("\nbeta manual:      ", np.round(modelo["beta"], 4))
    print("se manual:        ", np.round(modelo["se"], 4))

    Xc = sm.add_constant(X)
    m_sm = sm.Logit(y, Xc).fit(disp=0)  # sm usa Newton por default
    print("\nbeta statsmodels: ", np.round(np.asarray(m_sm.params), 4))
    print("se statsmodels:   ", np.round(np.asarray(m_sm.bse), 4))

    print(
        "\n=== 2. Box-Tidwell manual (x2 deberia marcar violacion: la verdad usa log(x2)) ==="
    )
    df_test = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    box_tidwell_manual(
        pd.Series.to_frame(df_test["x1"]).join(df_test[["x2", "x3"]]),
        pd.Series(y),
        cols_testear=["x2"],
        cols_control=["x1", "x3"],
    )
