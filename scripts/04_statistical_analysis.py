"""
04_statistical_analysis.py — Escalado ×2, Score Total y estadística MW
========================================================================
Requiere en la misma carpeta results/:
  - resultados_globales.csv     (generado por los scripts 01-03)
  - puntuacion_fisioterapia.csv (puntuaciones clínicas del fisioterapeuta)

1. Carga ambos CSV y une las puntuaciones clínicas al global
2. Escala los scores instrumentales ×2 → escala 0–2 por test
3. Calcula Inst_Score_Total = suma de los tres tests → escala 0–6
4. Test de Mann-Whitney U bilateral: VIP vs. sano para cada test
   (scores clínicos del fisioterapeuta + scores instrumentales)
5. Guarda resultados_globales.csv actualizado y tabla estadística en
   results/estadistica_MW.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

# Forzar UTF-8 en la consola Windows para evitar UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Rutas ─────────────────────────────────────────────────────────────────────

RESULTS_PATH = r'C:\Users\marti\Desktop\TFG\results\resultados_globales.csv'
CLINICAL_PATH = r'C:\Users\marti\Desktop\TFG\results\puntuacion_fisioterapia.csv'
STATS_PATH    = r'C:\Users\marti\Desktop\TFG\results\estadistica_MW.csv'

# Columnas instrumentales originales (escala 0–1) → nombre ×2
INSTR_MAP = {
    'Test3_Score': 'Inst_Test3_2',    # media de T3A+T3B
    'T7_Score':    'Inst_Test7_2',
    'T11_Score':   'Inst_Test11_2',
}


# ── 2. Parsear puntuaciones clínicas ─────────────────────────────────────────

def load_clinical_scores():
    """
    Extrae Test3a, Test3b, Test7, Test11 del CSV del fisioterapeuta.
    El CSV usa sep=';' con nombres de fila B01-B20 y H01-H20.

    Índices de columna (0-based, contando desde la columna de sujeto):
      0 → sujeto  |  3 → Test3a  |  4 → Test3b
     15 → Test7   | 20 → Test11
    """
    COL_SUBJ = 0
    COL_T3A  = 3
    COL_T3B  = 4
    COL_T7   = 15
    COL_T11  = 20

    raw = pd.read_csv(CLINICAL_PATH, sep=';', header=None,
                      dtype=str, encoding='utf-8-sig')

    def to_float(val):
        v = str(val).strip().replace(',', '.')
        try:
            return float(v)
        except ValueError:
            return np.nan

    rows = []
    for _, row in raw.iterrows():
        subj_raw = str(row.iloc[COL_SUBJ]).strip()

        # Solo filas B01-B20 / H01-H20
        if not (subj_raw.upper().startswith('B') or
                subj_raw.upper().startswith('H')):
            continue
        try:
            num = int(subj_raw[1:])
        except ValueError:
            continue
        if not (1 <= num <= 20):
            continue

        subject = subj_raw.lower()          # b01, h01, …
        group   = 'blind' if subject.startswith('b') else 'healthy'

        t3a = to_float(row.iloc[COL_T3A])
        t3b = to_float(row.iloc[COL_T3B])
        t7  = to_float(row.iloc[COL_T7])
        t11 = to_float(row.iloc[COL_T11])

        # Media Test3 clínico (NaN si ambas NaN)
        avail = [v for v in (t3a, t3b) if not np.isnan(v)]
        t3_med = float(np.mean(avail)) if avail else np.nan

        rows.append(dict(
            subject        = subject,
            group          = group,
            Clin_Test3a    = t3a,
            Clin_Test3b    = t3b,
            Clin_Test3_med = t3_med,
            Clin_Test7     = t7,
            Clin_Test11    = t11,
        ))

    return pd.DataFrame(rows)


# ── 3. Escalar instrumentales ×2 y calcular Score Total ───────────────────────

def scale_instrumental(df):
    """Añade columnas Inst_*_2 (escala 0–2) e Inst_Score_Total (0–6)."""
    for col_src, col_dst in INSTR_MAP.items():
        df[col_dst] = df[col_src] * 2.0 if col_src in df.columns else np.nan

    score_cols = list(INSTR_MAP.values())
    df['Inst_Score_Total'] = df[score_cols].sum(axis=1, min_count=1)
    return df


# ── 4. Test de Mann-Whitney U ─────────────────────────────────────────────────

def rank_biserial(u, n1, n2):
    """Correlación rank-biserial: r = 1 − 2U / (n1·n2)."""
    denom = n1 * n2
    if denom == 0:
        return np.nan
    return float(1.0 - 2.0 * u / denom)


def mw_test(df, col, group_col='group', g1='blind', g2='healthy'):
    """Mann-Whitney U bilateral entre dos grupos. Devuelve dict de resultados."""
    a = df.loc[df[group_col] == g1, col].dropna().values
    b = df.loc[df[group_col] == g2, col].dropna().values

    empty = dict(
        mean_blind=np.nan, sd_blind=np.nan, n_blind=len(a),
        mean_healthy=np.nan, sd_healthy=np.nan, n_healthy=len(b),
        U=np.nan, p=np.nan, r=np.nan, sig='',
    )
    if len(a) < 2 or len(b) < 2:
        return empty

    stat, p = mannwhitneyu(a, b, alternative='two-sided')
    r = rank_biserial(stat, len(a), len(b))

    return dict(
        mean_blind   = float(np.mean(a)),
        sd_blind     = float(np.std(a, ddof=1)),
        n_blind      = int(len(a)),
        mean_healthy = float(np.mean(b)),
        sd_healthy   = float(np.std(b, ddof=1)),
        n_healthy    = int(len(b)),
        U            = float(stat),
        p            = float(p),
        r            = float(r),
        sig          = '*' if p < 0.05 else '',
    )


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main():
    # --- a) Cargar puntuaciones clínicas ---
    if not os.path.exists(CLINICAL_PATH):
        print(f"[ERROR] No se encontró {CLINICAL_PATH}")
        print("        Coloca el archivo puntuacion_fisioterapia.csv en la carpeta results/")
        return

    df_clin = load_clinical_scores()
    print(f"✓ Puntuaciones clínicas cargadas: {len(df_clin)} sujetos "
          f"({(df_clin.group == 'blind').sum()} VIP, "
          f"{(df_clin.group == 'healthy').sum()} sanos)")

    # --- c) Cargar resultados instrumentales ---
    if not os.path.exists(RESULTS_PATH):
        print(f"\n[ERROR] No se encontró {RESULTS_PATH}")
        print("        Ejecuta primero 01_test3_analysis.py, "
              "02_test7_analysis.py y 03_test11_analysis.py")
        return

    df_g = pd.read_csv(RESULTS_PATH, sep=';', decimal=',')
    print(f"✓ resultados_globales.csv cargado: {len(df_g)} sujetos")

    # --- d) Eliminar columnas previas de este script para re-ejecución limpia ---
    drop = [c for c in df_g.columns
            if c.startswith('Clin_') or c.startswith('Inst_')]
    df_g = df_g.drop(columns=drop, errors='ignore')

    # --- e) Unir puntuaciones clínicas ---
    df_g = df_g.merge(df_clin.drop(columns='group'), on='subject', how='left')

    # --- f) Escalar ×2 y calcular total ---
    df_g = scale_instrumental(df_g)

    # --- g) Guardar CSV actualizado ---
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    df_g.to_csv(RESULTS_PATH, index=False, sep=';', decimal=',')
    print(f"✓ resultados_globales.csv actualizado → {RESULTS_PATH}")

    # ── Tabla de estadística ──────────────────────────────────────────────────
    tests_info = [
        # (etiqueta,    columna clínica,    columna instrumental ×2)
        ('Test 3a',   'Clin_Test3a',    None),
        ('Test 3b',   'Clin_Test3b',    None),
        ('Test 3',    'Clin_Test3_med', 'Inst_Test3_2'),
        ('Test 7',    'Clin_Test7',     'Inst_Test7_2'),
        ('Test 11',   'Clin_Test11',    'Inst_Test11_2'),
        ('Total 0-6', None,             'Inst_Score_Total'),
    ]

    hdr = (f"{'Test':<12} {'Tipo':<14} "
           f"{'n VIP':>6} {'Media VIP':>10} {'DE VIP':>8}  "
           f"{'n Sano':>6} {'Media Sano':>10} {'DE Sano':>8}  "
           f"{'U':>8} {'p-valor':>9} {'r':>6}  {'':>1}")
    sep = '─' * len(hdr)

    print(f"\n{sep}\n{hdr}\n{sep}")

    stat_rows = []
    for label, col_clin, col_inst in tests_info:
        for tipo, col in [('Clínico', col_clin), ('Instrumental', col_inst)]:
            if col is None or col not in df_g.columns:
                continue
            res = mw_test(df_g, col)
            sig = res['sig']

            # Formatear con manejo de NaN
            def fmt(x, fmt_str='.3f'):
                return format(x, fmt_str) if not np.isnan(x) else '  —   '

            print(
                f"{label:<12} {tipo:<14} "
                f"{res['n_blind']:>6} {fmt(res['mean_blind']):>10} {fmt(res['sd_blind']):>8}  "
                f"{res['n_healthy']:>6} {fmt(res['mean_healthy']):>10} {fmt(res['sd_healthy']):>8}  "
                f"{fmt(res['U'],'.1f'):>8} {fmt(res['p'],'.4f'):>9} {fmt(res['r']):>6}  {sig:>1}"
            )

            stat_rows.append(dict(
                Test=label, Tipo=tipo, Columna=col,
                **{k: res[k] for k in res if k != 'sig'},
                Significativo=sig,
            ))

    print(sep)
    print("* p < 0.05  |  r: rank-biserial (|r|>0.3 efecto medio, |r|>0.5 grande)\n")

    # --- h) Guardar tabla estadística ---
    df_stats = pd.DataFrame(stat_rows)
    df_stats.to_csv(STATS_PATH, index=False, sep=';', decimal=',')
    print(f"✓ Tabla estadística guardada → {STATS_PATH}")

    # ── Resumen escalado instrumental por sujeto ──────────────────────────────
    show_cols = ['subject', 'group',
                 'Inst_Test3_2', 'Inst_Test7_2', 'Inst_Test11_2',
                 'Inst_Score_Total']
    show_cols = [c for c in show_cols if c in df_g.columns]
    print("\nPuntuaciones instrumentales escaladas (0–2 por test, 0–6 total):")
    print(df_g[show_cols].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
