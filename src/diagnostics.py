import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. Empirical logit plot
# ---------------------------------------------------------------------------

def empirical_logit_plot(X_col, y, n_bins=10, save_dir=None):
    """
    Grafica logit empirico vs media del bin.
    Si save_dir es un Path, guarda la imagen ahi.
    """
    df = pd.DataFrame({"X": X_col.astype("float32"), "y": y})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    df["bin"] = pd.qcut(df["X"], q=n_bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True).agg(
        p=("y", "mean"),
        X_mid=("X", "mean"),
    )
    grouped = grouped[(grouped["p"] > 0) & (grouped["p"] < 1)]

    if grouped.empty:
        print(f"[skip] {X_col.name}: no hay bins validos para graficar")
        return

    grouped["logit"] = np.log(grouped["p"] / (1 - grouped["p"]))

    plt.figure()
    plt.scatter(grouped["X_mid"], grouped["logit"])
    xs = np.unique(grouped["X_mid"])
    coef = np.polyfit(grouped["X_mid"], grouped["logit"], 1)
    plt.plot(xs, np.poly1d(coef)(xs), "r--")
    plt.xlabel(X_col.name)
    plt.ylabel("logit empirico")
    plt.title(f"Logit empirico vs {X_col.name}")

    if save_dir is not None:
        plt.savefig(Path(save_dir) / f"linealidad-logit-{X_col.name}.png")

    plt.show()


# ---------------------------------------------------------------------------
# 2. Box-Tidwell: Wald por termino + LRT global
# ---------------------------------------------------------------------------

def box_tidwell_test(df, y, cols_a_testear, cols_control=None):
    """
    H0: gamma_j = 0 (linealidad en logit) para cada continua > 0.

    cols_a_testear : continuas positivas a testear.
    cols_control   : variables de control que entran al modelo pero no se testean.

    Retorna (modelo_aumentado, modelo_reducido, dict de resultados por variable).
    """
    X = pd.DataFrame(index=df.index)

    if cols_control:
        for c in cols_control:
            X[c] = df[c]

    interaction_cols = []
    for c in cols_a_testear:
        v = df[c]
        if not (v > 0).all():
            print(f"[omitida] {c}: tiene valores <= 0, aplica flag o desplazamiento primero")
            continue
        X[c] = v
        X[f"{c}_ln"] = v * np.log(v)
        interaction_cols.append(f"{c}_ln")

    X["_y"] = y.values
    X = X.dropna()
    y_clean = X.pop("_y")

    Xc = sm.add_constant(X)
    m_full = sm.Logit(y_clean, Xc).fit(disp=0)

    print("== Box-Tidwell: Wald por termino (H0: gamma=0, linealidad) ==")
    resultados = {}
    for col in interaction_cols:
        var = col.replace("_ln", "")
        g = m_full.params[col]
        se = m_full.bse[col]
        z = m_full.tvalues[col]
        p = m_full.pvalues[col]
        # Con n grande p<0.05 puede ser trivial: mirar magnitud de gamma
        flag = "*** VIOLACION" if (p < 0.05 and abs(g) >= 1e-2) else "OK"
        print(f"  {var:25s}: gamma={g:+.5f}  z={z:7.3f}  p={p:.4f}  {flag}")
        resultados[var] = {"gamma": g, "se": se, "z": z, "p": p}

    # LRT global: H0 todos los gamma = 0
    m_red = sm.Logit(y_clean, Xc.drop(columns=interaction_cols)).fit(disp=0)
    lr = 2 * (m_full.llf - m_red.llf)
    gl = len(interaction_cols)
    p_lrt = stats.chi2.sf(lr, df=gl)
    print(f"\n== LRT global: G2={lr:.3f}, gl={gl}, p={p_lrt:.4f} ==")

    return m_full, m_red, resultados


# ---------------------------------------------------------------------------
# 3. Decision post Box-Tidwell y aplicacion de remedios
# ---------------------------------------------------------------------------

def aplicar_remedios(df, resultados, gamma_threshold=1e-2):
    """
    Aplica transformaciones segun resultado de box_tidwell_test.
    Retorna df modificado y lista de columnas finales para el modelo.
    """
    df = df.copy()
    cols_finales = []

    for var, r in resultados.items():
        if r["p"] >= 0.05 or abs(r["gamma"]) < gamma_threshold:
            cols_finales.append(var)
        else:
            print(f"{var}: VIOLACION real -> agregando termino cuadratico")
            df[f"{var}_sq"] = df[var] ** 2
            cols_finales += [var, f"{var}_sq"]

    return df, cols_finales


# ---------------------------------------------------------------------------
# 4. VIF: multicolinealidad
# ---------------------------------------------------------------------------

def vif_report(X_df):
    """
    Calcula VIF para cada columna de X_df.
    VIF > 10 indica multicolinealidad problematica.
    """
    X = X_df.dropna()
    vif_data = pd.DataFrame({
        "variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
    }).sort_values("VIF", ascending=False)

    print("== VIF (> 10 indica multicolinealidad problematica) ==")
    print(vif_data.to_string(index=False))
    return vif_data
