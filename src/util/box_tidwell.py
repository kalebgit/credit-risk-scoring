import statsmodels.api as sm
import numpy as np


def box_tidwell_test(X_df, y):
    """
    Agrega términos X*ln(X) al modelo y testea si sus coeficientes
    son significativos (H0: gamma=0, i.e. linealidad en logit).
    """
    X_aug = X_df.copy()
    interaction_cols = []

    for col in X_df.columns:
        vals = X_df[col]
        # Solo para variables positivas (ln requiere X > 0)
        if (vals > 0).all():
            X_aug[f"{col}_ln"] = vals * np.log(vals)
            interaction_cols.append(f"{col}_ln")
        else:
            print(f"{col}: min={X_df[col].min():.3f}  → necesita desplazamiento")

    X_const = sm.add_constant(X_aug)
    model = sm.Logit(y, X_const).fit(disp=0)

    print("Test de Box-Tidwell: coeficientes de términos X*ln(X)")
    print("H0: coeficiente = 0 (linealidad). p < 0.05 => violación")
    print()
    for col in interaction_cols:
        coef = model.params[col]
        pval = model.pvalues[col]
        flag = "*** VIOLACIÓN" if pval < 0.05 else "OK"
        print(f"  {col:20s}: coef={coef:.4f}, p={pval:.4f}  {flag}")

    return model
