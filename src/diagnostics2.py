import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats
from sqlalchemy import create_engine
from pathlib import Path
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------------------------------------------------------------------------
# Conexion
# ---------------------------------------------------------------------------
db_user = os.environ["DB_USER"]
db_password = os.environ["DB_PASSWORD"]
db_name = os.environ["DB_NAME"]
db_port = os.environ["DB_PORT"]
engine = create_engine(f"postgresql+psycopg://{db_user}:{db_password}@localhost:{db_port}/{db_name}")

print("Cargando datos...")
df = pd.read_sql("SELECT * FROM staging.application_train", con=engine)
print(f"  Shape: {df.shape}")

output_dir = Path(__file__).parent.parent / "images_output"
output_dir.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def empirical_logit_plot(X_col, y, n_bins=10, save_dir=None, show=False):
    tmp = pd.DataFrame({"X": X_col.astype("float32"), "y": y})
    tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
    tmp["bin"] = pd.qcut(tmp["X"], q=n_bins, duplicates="drop")
    grouped = tmp.groupby("bin", observed=True).agg(p=("y","mean"), X_mid=("X","mean"))
    grouped = grouped[(grouped["p"] > 0) & (grouped["p"] < 1)]
    if grouped.empty:
        print(f"  [skip] {X_col.name}: no hay bins validos")
        return
    grouped["logit"] = np.log(grouped["p"] / (1 - grouped["p"]))
    fig, ax = plt.subplots()
    ax.scatter(grouped["X_mid"], grouped["logit"])
    xs = np.unique(grouped["X_mid"])
    coef = np.polyfit(grouped["X_mid"], grouped["logit"], 1)
    ax.plot(xs, np.poly1d(coef)(xs), "r--")
    ax.set_xlabel(X_col.name); ax.set_ylabel("logit empirico")
    ax.set_title(f"Logit empirico vs {X_col.name}")
    if save_dir:
        fig.savefig(Path(save_dir) / f"logit-{X_col.name}.png")
    plt.close(fig)


def box_tidwell(df, y, cols_testear, cols_control):
    X = pd.DataFrame(index=df.index)
    for c in cols_control:
        if c in df.columns:
            X[c] = df[c]
    interaction_cols = []
    for c in cols_testear:
        if c not in df.columns:
            continue
        v = df[c].astype("float64")
        if not (v > 0).all():
            print(f"  [omitida] {c}: tiene valores <= 0")
            continue
        X[c] = v
        X[f"{c}_ln"] = v * np.log(v)
        interaction_cols.append(f"{c}_ln")
    if not interaction_cols:
        print("  No hay columnas validas para Box-Tidwell")
        return None, None, {}
    X["_y"] = y.values
    X = X.dropna()
    y_clean = X.pop("_y")

    # estandarizar para evitar matriz singular
    X_std = (X - X.mean()) / X.std().replace(0, 1)
    Xc = sm.add_constant(X_std)

    fit_kwargs = dict(disp=0, method="bfgs", maxiter=200)
    m_full = sm.Logit(y_clean, Xc).fit(**fit_kwargs)
    resultados = {}
    for col in interaction_cols:
        var = col.replace("_ln", "")
        g = m_full.params[col]; se = m_full.bse[col]
        z = m_full.tvalues[col]; p = m_full.pvalues[col]
        flag = "*** VIOLACION" if (p < 0.05 and abs(g) >= 1e-2) else "OK"
        print(f"  {var:30s}: gamma={g:+.5f}  z={z:7.3f}  p={p:.4f}  {flag}")
        resultados[var] = {"gamma": g, "se": se, "z": z, "p": p}
    m_red = sm.Logit(y_clean, Xc.drop(columns=interaction_cols)).fit(**fit_kwargs)
    lr = 2 * (m_full.llf - m_red.llf)
    gl = len(interaction_cols)
    p_lrt = stats.chi2.sf(lr, df=gl)
    print(f"\n  LRT global: G2={lr:.3f}, gl={gl}, p={p_lrt:.4f}")
    return m_full, m_red, resultados


def vif_report(X_df):
    X = X_df.dropna().select_dtypes("number")
    # quitar columnas con varianza 0
    X = X.loc[:, X.std() > 0]
    vif = pd.DataFrame({
        "variable": X.columns,
        "VIF": [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    }).sort_values("VIF", ascending=False)
    print(vif.to_string(index=False))
    return vif

# ---------------------------------------------------------------------------
# 1. Preparacion inicial (igual que notebook)
# ---------------------------------------------------------------------------
print("\n=== PREPARACION ===")
df_t = df.copy()
nunique = df_t.nunique()

# float16 -> float32
f16 = df_t.select_dtypes("float16").columns
df_t[f16] = df_t[f16].astype("float32")

# cnt_children categorica
df_t["cnt_children"] = df_t["cnt_children"].clip(upper=4)

# dummies
cat_str = df_t[nunique[nunique <= 5].index].select_dtypes(include="str").columns
cat_num = ["cnt_children"]
cols_dummy = list(set(cat_num) | set(cat_str))
df_t = pd.get_dummies(df_t, columns=cols_dummy, drop_first=True)
# get_dummies puede dejar bool; convertir todo a int
bool_cols = df_t.select_dtypes("bool").columns
df_t[bool_cols] = df_t[bool_cols].astype(int)

# valores especiales
SPECIAL = 365243
df_t["flag_current_employee"] = (df_t["days_employed"] != SPECIAL).astype(int)
df_t["days_employed"] = df_t["days_employed"].replace(SPECIAL, np.nan)

df_t["flag_goods_type"] = (df_t["amt_goods_price"] > 0).astype(int)
df_t["amt_goods_price"] = df_t["amt_goods_price"].replace(0, np.nan)

y = df_t["target"].astype(int)

# columnas continuas (> 2 valores unicos, numericas, sin target)
nunique_t = df_t.nunique()
non_binary = nunique_t[nunique_t > 2].index
continuous_cols = list(df_t[non_binary].select_dtypes("number").columns)
continuous_cols = [c for c in continuous_cols if c != "target"]
print(f"Continuas iniciales: {len(continuous_cols)}")

# ---------------------------------------------------------------------------
# 2. Empirical logit plots iniciales
# ---------------------------------------------------------------------------
print("\n=== EMPIRICAL LOGIT PLOTS (inicial) ===")
for col in continuous_cols:
    mask = df_t[col].notna()
    empirical_logit_plot(df_t.loc[mask, col], y[mask], save_dir=output_dir)
print(f"  Guardadas en {output_dir}")

# ---------------------------------------------------------------------------
# 3. Transformaciones log para montos de escala amplia
# ---------------------------------------------------------------------------
print("\n=== TRANSFORMACIONES LOG ===")
log_candidates = ["amt_income_total", "amt_credit", "amt_annuity", "amt_goods_price"]
transformaciones = {}

for col in log_candidates:
    if col in df_t.columns:
        new_col = f"log_{col}"
        vals = df_t[col]
        # solo aplicar log donde el valor es positivo; el resto queda NaN
        df_t[new_col] = np.where(vals > 0, np.log(vals.where(vals > 0)), np.nan)
        if col in continuous_cols:
            continuous_cols.remove(col)
        continuous_cols.append(new_col)
        transformaciones[col] = f"log -> {new_col}"
        print(f"  {col} -> {new_col}")

# re-graficar transformadas
print("  Re-graficando transformadas...")
for col in [f"log_{c}" for c in log_candidates if f"log_{c}" in df_t.columns]:
    mask = df_t[col].notna()
    empirical_logit_plot(df_t.loc[mask, col], y[mask], save_dir=output_dir)

# ---------------------------------------------------------------------------
# 4. Box-Tidwell iterativo
# ---------------------------------------------------------------------------
print("\n=== BOX-TIDWELL ITERACION 1 ===")

# solo continuas positivas (candidatas al test)
cols_testear = [c for c in continuous_cols if df_t[c].dropna().gt(0).all()]
# dummies y flags como control (solo numericas, no las continuas ni target)
cols_control = [c for c in df_t.select_dtypes("number").columns
                if c not in continuous_cols + ["target", "sk_id_curr"]]

print(f"  Columnas a testear: {len(cols_testear)}")
print(f"  Columnas control:   {len(cols_control)}")

m_full, m_red, resultados = box_tidwell(df_t, y, cols_testear, cols_control)

# ---------------------------------------------------------------------------
# 5. Aplicar remedios segun resultado
# ---------------------------------------------------------------------------
print("\n=== APLICANDO REMEDIOS ===")
remedios_aplicados = {}

for var, r in resultados.items():
    if r["p"] >= 0.05 or abs(r["gamma"]) < 1e-2:
        print(f"  {var}: OK")
    else:
        print(f"  {var}: VIOLACION real -> termino cuadratico")
        df_t[f"{var}_sq"] = df_t[var] ** 2
        if var in continuous_cols:
            continuous_cols.append(f"{var}_sq")
        remedios_aplicados[var] = f"+ {var}_sq"
        transformaciones[var] = f"cuadratico -> {var}_sq"

# ---------------------------------------------------------------------------
# 6. Box-Tidwell iteracion 2 (si hubo remedios)
# ---------------------------------------------------------------------------
if remedios_aplicados:
    print("\n=== BOX-TIDWELL ITERACION 2 ===")
    cols_testear2 = [c for c in continuous_cols if df_t[c].dropna().gt(0).all()]
    m_full2, m_red2, resultados2 = box_tidwell(df_t, y, cols_testear2, cols_control)
    resultados_finales = resultados2
else:
    print("\n  No se aplicaron remedios, resultado de iteracion 1 es final")
    resultados_finales = resultados

# ---------------------------------------------------------------------------
# 7. VIF
# ---------------------------------------------------------------------------
print("\n=== VIF (multicolinealidad) ===")
X_vif = df_t[continuous_cols].copy()
vif_df = vif_report(X_vif)

# ---------------------------------------------------------------------------
# 8. Resumen final
# ---------------------------------------------------------------------------
print("\n=== RESUMEN FINAL ===")
print("\nTransformaciones aplicadas:")
for orig, trans in transformaciones.items():
    print(f"  {orig:30s} -> {trans}")

print("\nResultado Box-Tidwell por variable:")
violaciones = []
for var, r in resultados_finales.items():
    cumple = r["p"] >= 0.05 or abs(r["gamma"]) < 1e-2
    estado = "OK" if cumple else "VIOLACION"
    print(f"  {var:30s}: p={r['p']:.4f}  |gamma|={abs(r['gamma']):.5f}  {estado}")
    if not cumple:
        violaciones.append(var)

print(f"\nVIF problematicos (>10):")
prob_vif = vif_df[vif_df["VIF"] > 10]
if prob_vif.empty:
    print("  Ninguno")
else:
    print(prob_vif.to_string(index=False))

if not violaciones:
    print("\n✓ Hipotesis de linealidad en el logit: SE CUMPLE para todas las variables")
else:
    print(f"\n✗ Aun con violaciones en: {violaciones}")
    print("  Considerar spline cubico o excluir la variable")

print("\nColumnas finales para el modelo:")
print(continuous_cols)
