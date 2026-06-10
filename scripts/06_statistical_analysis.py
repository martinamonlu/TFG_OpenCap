"""
04_statistical_analysis.py — Escalado ×2, Score Total y estadística MW + Spearman
===================================================================================
Requiere en la misma carpeta results/:
  - resultados_globales.csv     (generado por los scripts 01-03)
  - puntuacion_fisioterapia.csv (puntuaciones clínicas del fisioterapeuta)

1. Carga ambos CSV y une las puntuaciones clínicas al global
2. Escala los scores instrumentales ×2 → escala 0–2 por test
3. Calcula Inst_Score_Total = suma de los tres tests → escala 0–6
4. Test de Mann-Whitney U: VIP vs. sano (mediana ± IQR, U, p, r rank-biserial)
5. Shapiro-Wilk por grupo para justificar el test no paramétrico
6. Correlación de Spearman entre score instrumentado y puntuación clínica
7. Guarda resultados_globales.csv actualizado y tabla estadística en
   results/estadistica_MW.csv
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, shapiro

# Forzar UTF-8 en la consola Windows para evitar UnicodeEncodeError
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── Rutas ─────────────────────────────────────────────────────────────────────

RESULTS_PATH  = r'C:\Users\marti\Desktop\TFG\results\resultados_globales.csv'
CLINICAL_PATH = r'C:\Users\marti\Desktop\TFG\results\puntuacion_fisioterapia.csv'
STATS_PATH    = r'C:\Users\marti\Desktop\TFG\results\estadistica_MW.csv'

# Columnas instrumentales originales (escala 0–1) → nombre ×2
INSTR_MAP = {
    'Test3_Score': 'Inst_Test3_2',
    'T7_Score':    'Inst_Test7_2',
    'T11_Score':   'Inst_Test11_2',
}


# ── Parsear puntuaciones clínicas ─────────────────────────────────────────────

def load_clinical_scores():
    """
    Extrae Test3a, Test3b, Test7, Test11 del CSV del fisioterapeuta.
    Índices de columna (0-based):
      0 → sujeto  |  3 → Test3a  |  4 → Test3b  |  15 → Test7  |  20 → Test11
    """
    COL_SUBJ = 0; COL_T3A = 3; COL_T3B = 4; COL_T7 = 15; COL_T11 = 20

    raw = pd.read_csv(CLINICAL_PATH, sep=';', header=None,
                      dtype=str, encoding='utf-8-sig')

    def to_float(val):
        v = str(val).strip().replace(',', '.')
        try:    return float(v)
        except: return np.nan

    rows = []
    for _, row in raw.iterrows():
        subj_raw = str(row.iloc[COL_SUBJ]).strip()
        if not (subj_raw.upper().startswith('B') or subj_raw.upper().startswith('H')):
            continue
        try:    num = int(subj_raw[1:])
        except: continue
        if not (1 <= num <= 20):
            continue

        subject = subj_raw.lower()
        group   = 'blind' if subject.startswith('b') else 'healthy'
        t3a = to_float(row.iloc[COL_T3A])
        t3b = to_float(row.iloc[COL_T3B])
        t7  = to_float(row.iloc[COL_T7])
        t11 = to_float(row.iloc[COL_T11])
        avail  = [v for v in (t3a, t3b) if not np.isnan(v)]
        t3_med = float(np.mean(avail)) if avail else np.nan

        rows.append(dict(subject=subject, group=group,
                         Clin_Test3a=t3a, Clin_Test3b=t3b,
                         Clin_Test3_med=t3_med, Clin_Test7=t7, Clin_Test11=t11))

    return pd.DataFrame(rows)


# ── Escalar instrumentales ×2 y calcular Score Total ─────────────────────────

def scale_instrumental(df):
    """Añade columnas Inst_*_2 (escala 0–2) e Inst_Score_Total (0–6)."""
    for col_src, col_dst in INSTR_MAP.items():
        df[col_dst] = df[col_src] * 2.0 if col_src in df.columns else np.nan
    score_cols = list(INSTR_MAP.values())
    df['Inst_Score_Total'] = df[score_cols].sum(axis=1, min_count=1)
    return df


# ── Mann-Whitney U ────────────────────────────────────────────────────────────

def rank_biserial(u, n1, n2):
    denom = n1 * n2
    return float(1.0 - 2.0 * u / denom) if denom > 0 else np.nan


def mw_test(df, col, group_col='group', g1='blind', g2='healthy'):
    """
    Mann-Whitney U bilateral entre dos grupos.
    Devuelve mediana, IQR, U, p y rank-biserial r.
    """
    a = df.loc[df[group_col] == g1, col].dropna().values
    b = df.loc[df[group_col] == g2, col].dropna().values

    empty = dict(med_blind=np.nan, iqr_blind=np.nan, n_blind=len(a),
                 med_healthy=np.nan, iqr_healthy=np.nan, n_healthy=len(b),
                 U=np.nan, p=np.nan, r=np.nan, sig='')
    if len(a) < 2 or len(b) < 2:
        return empty

    stat, p = mannwhitneyu(a, b, alternative='two-sided')
    r = rank_biserial(stat, len(a), len(b))

    def iqr(x):
        return float(np.percentile(x, 75) - np.percentile(x, 25))

    return dict(
        med_blind   = float(np.median(a)),
        iqr_blind   = iqr(a),
        n_blind     = int(len(a)),
        med_healthy = float(np.median(b)),
        iqr_healthy = iqr(b),
        n_healthy   = int(len(b)),
        U           = float(stat),
        p           = float(p),
        r           = float(r),
        sig         = ('***' if p < 0.001 else
                       '**'  if p < 0.01  else
                       '*'   if p < 0.05  else 'ns'),
    )


# ── Shapiro-Wilk ──────────────────────────────────────────────────────────────

def shapiro_test(df, col, group_col='group', g1='blind', g2='healthy'):
    """Shapiro-Wilk por grupo. Devuelve p-valores (NaN si n < 3 o n > 5000)."""
    result = {}
    for label, grp in [(g1, g1), (g2, g2)]:
        x = df.loc[df[group_col] == grp, col].dropna().values
        if 3 <= len(x) <= 5000:
            _, p = shapiro(x)
            result[f'sw_p_{label}'] = float(p)
        else:
            result[f'sw_p_{label}'] = np.nan
    return result


# ── Spearman ──────────────────────────────────────────────────────────────────

def spearman_test(df, col_inst, col_clin):
    """Correlación de Spearman entre score instrumentado y puntuación clínica."""
    merged = df[[col_inst, col_clin]].dropna()
    if len(merged) < 5:
        return dict(rho=np.nan, p=np.nan, n=len(merged), sig='', note='n insuficiente')
    if merged[col_clin].std() == 0:
        return dict(rho=np.nan, p=np.nan, n=len(merged), sig='',
                    note='Puntuacion clinica constante — Spearman no calculable')
    rho, p = spearmanr(merged[col_inst], merged[col_clin])
    return dict(
        rho  = float(rho),
        p    = float(p),
        n    = int(len(merged)),
        sig  = ('***' if p < 0.001 else
                '**'  if p < 0.01  else
                '*'   if p < 0.05  else 'ns'),
        note = '',
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # --- a) Cargar puntuaciones clínicas ---
    if not os.path.exists(CLINICAL_PATH):
        print(f"[ERROR] No se encontró {CLINICAL_PATH}")
        print("        Coloca puntuacion_fisioterapia.csv en la carpeta results/")
        return

    df_clin = load_clinical_scores()
    print(f"✓ Puntuaciones clínicas cargadas: {len(df_clin)} sujetos "
          f"({(df_clin.group == 'blind').sum()} VIP, "
          f"{(df_clin.group == 'healthy').sum()} sanos)")

    # --- b) Cargar resultados instrumentales ---
    if not os.path.exists(RESULTS_PATH):
        print(f"\n[ERROR] No se encontró {RESULTS_PATH}")
        print("        Ejecuta primero 01_test3_analysis.py, "
              "02_test7_analysis.py y 03_test11_analysis.py")
        return

    df_g = pd.read_csv(RESULTS_PATH, sep=';', decimal=',')
    print(f"✓ resultados_globales.csv cargado: {len(df_g)} sujetos")

    # --- c) Limpiar columnas previas de este script ---
    drop = [c for c in df_g.columns if c.startswith('Clin_') or c.startswith('Inst_')]
    df_g = df_g.drop(columns=drop, errors='ignore')

    # --- d) Unir puntuaciones clínicas y escalar ---
    df_g = df_g.merge(df_clin.drop(columns='group'), on='subject', how='left')
    df_g = scale_instrumental(df_g)

    # --- e) Guardar CSV actualizado ---
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    df_g.to_csv(RESULTS_PATH, index=False, sep=';', decimal=',')
    print(f"✓ resultados_globales.csv actualizado → {RESULTS_PATH}")

    # ── tests_info: (etiqueta, col_clin, col_inst) ────────────────────────────
    # col_clin se usa para Mann-Whitney clínico y para Spearman vs instrumental
    tests_info = [
        ('Test 3a',   'Clin_Test3a',    None),
        ('Test 3b',   'Clin_Test3b',    None),
        ('Test 3',    'Clin_Test3_med', 'Inst_Test3_2'),
        ('Test 7',    'Clin_Test7',     'Inst_Test7_2'),
        ('Test 11',   'Clin_Test11',    'Inst_Test11_2'),
        ('Total 0-6', None,             'Inst_Score_Total'),
    ]

    # ── Cabecera Mann-Whitney ─────────────────────────────────────────────────
    hdr_mw = (f"{'Test':<12} {'Tipo':<14} "
              f"{'n VIP':>6} {'Med VIP':>9} {'IQR VIP':>9}  "
              f"{'n Sano':>6} {'Med Sano':>9} {'IQR Sano':>9}  "
              f"{'U':>7} {'p':>8} {'r':>6}  {'sig':>3}")
    sep = '─' * len(hdr_mw)

    print(f"\n{'═'*len(hdr_mw)}")
    print("MANN-WHITNEY U  —  VIP vs. Control")
    print(f"{'═'*len(hdr_mw)}\n{hdr_mw}\n{sep}")

    stat_rows = []

    def fmt(x, fs='.3f'):
        return format(x, fs) if (x is not None and not np.isnan(x)) else '   —  '

    for label, col_clin, col_inst in tests_info:
        for tipo, col in [('Clinico', col_clin), ('Instrumental', col_inst)]:
            if col is None or col not in df_g.columns:
                continue

            res = mw_test(df_g, col)
            sw  = shapiro_test(df_g, col)

            print(
                f"{label:<12} {tipo:<14} "
                f"{res['n_blind']:>6} {fmt(res['med_blind']):>9} {fmt(res['iqr_blind']):>9}  "
                f"{res['n_healthy']:>6} {fmt(res['med_healthy']):>9} {fmt(res['iqr_healthy']):>9}  "
                f"{fmt(res['U'],'.1f'):>7} {fmt(res['p'],'.4f'):>8} {fmt(res['r']):>6}  "
                f"{res['sig']:>3}"
                f"   [SW: VIP p={fmt(sw['sw_p_blind'],'.3f')} "
                f"Sano p={fmt(sw['sw_p_healthy'],'.3f')}]"
            )

            stat_rows.append(dict(Test=label, Tipo=tipo, Columna=col,
                                  **{k: res[k] for k in res if k != 'sig'},
                                  Sig_MW=res['sig'],
                                  **sw))

    print(sep)
    print("ns p≥0.05  * p<0.05  ** p<0.01  *** p<0.001  |  "
          "r rank-biserial: |r|>0.3 medio, |r|>0.5 grande")
    print("SW: Shapiro-Wilk (p<0.05 → rechaza normalidad → MW justificado)\n")

    # ── Spearman: instrumental vs. clínico ────────────────────────────────────
    spearman_pairs = [
        ('Test 3',  'Inst_Test3_2',  'Clin_Test3_med'),
        ('Test 7',  'Inst_Test7_2',  'Clin_Test7'),
        ('Test 11', 'Inst_Test11_2', 'Clin_Test11'),
    ]

    hdr_sp = (f"{'Test':<12} {'n':>5} {'rho':>7} {'p':>8}  {'sig':>3}  {'Nota'}")
    sep_sp = '─' * 55

    print(f"{'═'*55}")
    print("SPEARMAN  —  Score instrumentado vs. puntuación clínica")
    print(f"{'═'*55}\n{hdr_sp}\n{sep_sp}")

    spearman_rows = []
    for label, col_inst, col_clin in spearman_pairs:
        if col_inst not in df_g.columns or col_clin not in df_g.columns:
            continue
        sp = spearman_test(df_g, col_inst, col_clin)
        print(f"{label:<12} {sp['n']:>5} {fmt(sp['rho']):>7} "
              f"{fmt(sp['p'],'.4f'):>8}  {sp['sig']:>3}  {sp['note']}")
        spearman_rows.append(dict(Test=label, Col_Inst=col_inst,
                                  Col_Clin=col_clin, **sp))

    print(sep_sp)
    print("rho: correlación de rangos de Spearman\n")

    # --- Guardar tabla estadística ---
    df_stats = pd.DataFrame(stat_rows)
    df_sp    = pd.DataFrame(spearman_rows)
    df_stats.to_csv(STATS_PATH, index=False, sep=';', decimal=',')
    sp_path = STATS_PATH.replace('estadistica_MW', 'estadistica_Spearman')
    df_sp.to_csv(sp_path, index=False, sep=';', decimal=',')
    print(f"✓ Tabla MW guardada       → {STATS_PATH}")
    print(f"✓ Tabla Spearman guardada → {sp_path}")

    # ── Resumen scores instrumentales por sujeto ──────────────────────────────
    show_cols = ['subject', 'group',
                 'Inst_Test3_2', 'Inst_Test7_2', 'Inst_Test11_2', 'Inst_Score_Total']
    show_cols = [c for c in show_cols if c in df_g.columns]
    print("\nPuntuaciones instrumentales escaladas (0–2 por test, 0–6 total):")
    print(df_g[show_cols].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
