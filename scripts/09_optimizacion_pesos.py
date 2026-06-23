"""
09_optimizacion_pesos.py - Optimizacion de los pesos de combinacion de mecanismos
==================================================================================
Da rigor a la eleccion de pesos del score instrumental: en vez de fijarlos "a
ojo", se buscan los que MAXIMIZAN la correlacion de Spearman entre el score y la
nota clinica del fisioterapeuta.

QUE se optimiza:
  Solo los pesos de la COMBINACION FINAL de mecanismos (los de arriba, p.ej.
  0.40/0.30/0.30). Los pesos INTERNOS de cada mecanismo se mantienen fijos,
  porque definen conceptualmente el mecanismo segun la literatura.

COMO (sin sesgo de circularidad):
  - Powell (derivative-free) porque Spearman no es derivable. Regularizacion L2
    (Ridge) para no inflar pesos con n pequeno.
  - Validacion cruzada 5-fold ESTRATIFICADA (mantiene la proporcion de grupos):
    los pesos se optimizan en 4 folds y la correlacion se evalua en el fold no
    visto -> correlacion HONESTA (no inflada por construccion).

Se comparan TRES correlaciones por test:
  rho_teorico   : pesos fijados a mano (no ajustados al fisio -> ya es honesta).
  rho_insample  : pesos optimizados sobre TODA la muestra (inflada, cota superior).
  rho_cv        : pesos optimizados con validacion cruzada (honesta).
Y la ESTABILIDAD de los pesos entre folds (mucha variabilidad -> sobreajuste).

Test 7 se excluye: la nota clinica es constante (Spearman no calculable).
Salida: results/optimizacion_pesos.csv (para el anexo).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedStratifiedKFold

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS = r'C:\Users\marti\Desktop\TFG\results\resultados_globales.csv'
OUT     = r'C:\Users\marti\Desktop\TFG\results\optimizacion_pesos.csv'

ALPHA   = 0.1      # fuerza de la regularizacion L2
SEED    = 0
N_REP   = 20       # repeticiones de la validacion cruzada 5-fold

# (nombre, columnas de mecanismos, col clinica, pesos teoricos, etiquetas)
TESTS = [
    ('Test 3a', ['T3A_S_Mec1', 'T3A_S_Mec2', 'T3A_S_Tiempo'], 'Clin_Test3a',
     [0.40, 0.30, 0.30], ['S_Mec1', 'S_Mec2', 'S_Tiempo']),
    ('Test 3b', ['T3B_S_Mec1', 'T3B_S_Mec2', 'T3B_S_Tiempo'], 'Clin_Test3b',
     [0.40, 0.30, 0.30], ['S_Mec1', 'S_Mec2', 'S_Tiempo']),
    ('Test 11', ['T11_S_Mec1', 'T11_S_Mec2', 'T11_S_Mec3'], 'Clin_Test11',
     [0.45, 0.35, 0.20], ['S_Mec1', 'S_Mec2', 'S_Mec3']),
    ('Test 14-1', ['T14_1_S_Tiempo', 'T14_1_S_Estabilidad'], 'Clin_Test14_1',
     [0.65, 0.35], ['S_Tiempo', 'S_Estab']),
    ('Test 14-2', ['T14_2_S_Tiempo', 'T14_2_S_Estabilidad'], 'Clin_Test14_2',
     [0.65, 0.35], ['S_Tiempo', 'S_Estab']),
]


def rho(score, y):
    r = spearmanr(score, y).correlation
    return 0.0 if np.isnan(r) else float(r)


def objetivo(w, X, y, alpha):
    """-Spearman + penalizacion L2. Los mecanismos son 'mayor = mejor', por eso
    los pesos se restringen a positivos."""
    return -rho(X @ w, y) + alpha * float(np.sum(w ** 2))


def optimizar(X, y, alpha=ALPHA):
    n = X.shape[1]
    res = minimize(objetivo, np.ones(n), args=(X, y, alpha),
                   method='Powell', bounds=[(0.0, 10.0)] * n)
    return res.x


def normaliza(w):
    s = float(np.sum(np.abs(w)))
    return w / s if s > 0 else w


def main():
    df = pd.read_csv(RESULTS, sep=';', decimal=',')
    df['grp'] = np.where(df.subject.str.startswith('h'), 'control', 'invidente')

    filas = []
    print("=" * 92)
    print("OPTIMIZACION DE PESOS (Powell + L2, validacion cruzada 5-fold estratificada)")
    print("=" * 92)

    for nombre, cols, clin, teoria, etiquetas in TESTS:
        sub = df.dropna(subset=cols + [clin])
        X = sub[cols].values
        y = sub[clin].values
        g = sub['grp'].values
        n = len(sub)
        teoria = np.array(teoria)

        # 1) Teorico (pesos a mano, NO ajustados al fisio -> honesto)
        rho_teo = rho(X @ teoria, y)

        # 2) Optimizado sobre TODA la muestra (in-sample, inflado)
        w_full = optimizar(X, y)
        rho_ins = rho(X @ w_full, y)
        w_full_n = normaliza(w_full)

        # 3) Validacion cruzada 5-fold estratificada (honesto)
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=N_REP, random_state=SEED)
        oof_sum = np.zeros(n); oof_cnt = np.zeros(n)
        w_folds = []
        for tr, te in cv.split(X, g):
            w = optimizar(X[tr], y[tr])
            oof_sum[te] += X[te] @ w
            oof_cnt[te] += 1
            w_folds.append(normaliza(w))
        oof = oof_sum / np.maximum(oof_cnt, 1)
        rho_cv = rho(oof, y)
        w_folds = np.array(w_folds)
        w_cv_mean = w_folds.mean(axis=0)
        w_cv_std = w_folds.std(axis=0)

        # -- Reporte --
        print(f"\n{nombre}  (n={n})")
        print(f"  {'mecanismo':<10}{'teorico':>9}{'optim(full)':>13}{'CV media+-sd':>18}")
        for i, et in enumerate(etiquetas):
            print(f"  {et:<10}{teoria[i]:>9.2f}{w_full_n[i]:>13.2f}"
                  f"{w_cv_mean[i]:>10.2f} +-{w_cv_std[i]:>4.2f}")
        print(f"  Spearman:  teorico={rho_teo:+.3f}   in-sample(inflado)={rho_ins:+.3f}"
              f"   CV(honesto)={rho_cv:+.3f}")

        fila = dict(Test=nombre, n=n, rho_teorico=round(rho_teo, 3),
                    rho_insample=round(rho_ins, 3), rho_cv=round(rho_cv, 3))
        for i, et in enumerate(etiquetas):
            fila[f'peso_teorico_{et}'] = round(float(teoria[i]), 3)
            fila[f'peso_optim_full_{et}'] = round(float(w_full_n[i]), 3)
            fila[f'peso_cv_media_{et}'] = round(float(w_cv_mean[i]), 3)
            fila[f'peso_cv_sd_{et}'] = round(float(w_cv_std[i]), 3)
        filas.append(fila)

    pd.DataFrame(filas).to_csv(OUT, index=False, sep=';', decimal=',')
    print("\n" + "=" * 92)
    print(f"[OK] Tabla guardada -> {OUT}")
    print("Lectura: si rho_cv ~ rho_teorico, los pesos a mano ya eran casi optimos.")
    print("Si rho_insample >> rho_cv, hay sobreajuste (la mejora era ilusoria).")
    print("Si los pesos CV tienen sd alta, son inestables -> poco fiables (n pequeno).")


if __name__ == '__main__':
    main()
