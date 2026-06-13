"""
06_statistical_analysis.py - Escalado x2, Score Total 0-12 y estadistica MW + Spearman
=======================================================================================
Requiere en results/:
  - resultados_globales.csv     (generado por los scripts 01-05)
  - puntuacion_fisioterapia.csv (puntuaciones clinicas del fisioterapeuta)

Cada test del Mini-BESTest se puntua 0-2. Aqui se tratan los 6 SUB-ENSAYOS
instrumentados por separado (Test 3a, 3b, 7, 11, 14-1, 14-2), cada uno escalado
x2 a la escala clinica 0-2:

  1. Carga ambos CSV y une las puntuaciones clinicas al global.
  2. Escala cada sub-ensayo instrumental x2 -> 0-2.
  3. Score Total = suma de los 6 sub-ensayos -> 0-12. Solo se calcula para
     sujetos con los 6 presentes (casos completos), para que los totales sean
     comparables entre sujetos. Se calcula tambien el total clinico 0-12.
  4. Mann-Whitney U: VIP vs sano (mediana, IQR, U, p, r rank-biserial).
  5. Shapiro-Wilk por grupo (justifica el test no parametrico).
  6. Spearman entre cada score instrumentado y su puntuacion clinica, y entre
     los totales 0-12 instrumental y clinico.
  7. Guarda resultados_globales.csv actualizado y las tablas estadisticas.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr, shapiro, kruskal

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -- Rutas --------------------------------------------------------------------

RESULTS_PATH  = r'C:\Users\marti\Desktop\TFG\results\resultados_globales.csv'
CLINICAL_PATH = r'C:\Users\marti\Desktop\TFG\results\puntuacion_fisioterapia.csv'
STATS_PATH    = r'C:\Users\marti\Desktop\TFG\results\estadistica_MW.csv'

# Los 6 sub-ensayos instrumentados: (etiqueta, col_origen, col_escalada, col_clinica)
# col_origen vive en resultados_globales (escala 0-1); col_escalada es x2 (0-2).
ITEMS = [
    ('Test 3a',   'T3A_Score',   'Inst_Test3a_2',   'Clin_Test3a'),
    ('Test 3b',   'T3B_Score',   'Inst_Test3b_2',   'Clin_Test3b'),
    ('Test 7',    'T7_Score',    'Inst_Test7_2',    'Clin_Test7'),
    ('Test 11',   'T11_Score',   'Inst_Test11_2',   'Clin_Test11'),
    ('Test 14-1', 'T14_1_Score', 'Inst_Test14_1_2', 'Clin_Test14_1'),
    ('Test 14-2', 'T14_2_Score', 'Inst_Test14_2_2', 'Clin_Test14_2'),
]


# -- Parsear puntuaciones clinicas --------------------------------------------

def load_clinical_scores():
    """
    Extrae las puntuaciones clinicas (0-2) de los 6 sub-ensayos.
    Indices de columna (0-based) en puntuacion_fisioterapia.csv:
      0 -> sujeto | 3 -> Test3a | 4 -> Test3b | 15 -> Test7 | 20 -> Test11
      23 -> Test14-1 | 24 -> Test14-2
    """
    COL = dict(subj=0, t3a=3, t3b=4, t7=15, t11=20, t14_1=23, t14_2=24)

    raw = pd.read_csv(CLINICAL_PATH, sep=';', header=None,
                      dtype=str, encoding='utf-8-sig')

    def to_float(val):
        v = str(val).strip().replace(',', '.')
        try:    return float(v)
        except: return np.nan

    rows = []
    for _, row in raw.iterrows():
        subj_raw = str(row.iloc[COL['subj']]).strip()
        if not (subj_raw.upper().startswith('B') or subj_raw.upper().startswith('H')):
            continue
        try:    num = int(subj_raw[1:])
        except: continue
        if not (1 <= num <= 20):
            continue

        subject = subj_raw.lower()
        group   = 'blind' if subject.startswith('b') else 'healthy'
        rows.append(dict(
            subject=subject, group=group,
            Clin_Test3a   = to_float(row.iloc[COL['t3a']]),
            Clin_Test3b   = to_float(row.iloc[COL['t3b']]),
            Clin_Test7    = to_float(row.iloc[COL['t7']]),
            Clin_Test11   = to_float(row.iloc[COL['t11']]),
            Clin_Test14_1 = to_float(row.iloc[COL['t14_1']]),
            Clin_Test14_2 = to_float(row.iloc[COL['t14_2']]),
        ))

    return pd.DataFrame(rows)


# -- Escalar instrumentales x2 y calcular totales 0-12 ------------------------

def build_scores(df):
    """
    Anade las columnas escaladas Inst_*_2 (0-2) de cada sub-ensayo y los dos
    totales 0-12 (instrumental y clinico) con regla de CASOS COMPLETOS: el
    total es NaN si al sujeto le falta cualquiera de los 6 sub-ensayos.
    """
    inst_scaled = []
    for _, src, scaled, _clin in ITEMS:
        df[scaled] = df[src] * 2.0 if src in df.columns else np.nan
        inst_scaled.append(scaled)

    clin_cols = [it[3] for it in ITEMS]

    def total_casos_completos(cols):
        presentes = [c for c in cols if c in df.columns]
        if len(presentes) < len(cols):
            return pd.Series(np.nan, index=df.index)
        completo = df[presentes].notna().all(axis=1)
        return pd.Series(np.where(completo, df[presentes].sum(axis=1), np.nan),
                         index=df.index)

    df['Inst_Score_Total'] = total_casos_completos(inst_scaled)
    df['Clin_Score_Total'] = total_casos_completos(clin_cols)
    return df


# -- Mann-Whitney U -----------------------------------------------------------

def rank_biserial(u, n1, n2):
    denom = n1 * n2
    return float(1.0 - 2.0 * u / denom) if denom > 0 else np.nan


def mw_test(df, col, group_col='group', g1='blind', g2='healthy'):
    """Mann-Whitney U bilateral. Devuelve mediana, IQR, U, p y rank-biserial r."""
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
        med_blind   = float(np.median(a)), iqr_blind = iqr(a), n_blind = int(len(a)),
        med_healthy = float(np.median(b)), iqr_healthy = iqr(b), n_healthy = int(len(b)),
        U = float(stat), p = float(p), r = float(r),
        sig = ('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'),
    )


# -- Shapiro-Wilk -------------------------------------------------------------

def shapiro_test(df, col, group_col='group', g1='blind', g2='healthy'):
    """Shapiro-Wilk por grupo. p-valor NaN si n < 3 o n > 5000."""
    result = {}
    for grp in (g1, g2):
        x = df.loc[df[group_col] == grp, col].dropna().values
        if 3 <= len(x) <= 5000:
            _, p = shapiro(x)
            result[f'sw_p_{grp}'] = float(p)
        else:
            result[f'sw_p_{grp}'] = np.nan
    return result


# -- Spearman -----------------------------------------------------------------

def spearman_test(df, col_inst, col_clin):
    """Spearman entre score instrumentado y puntuacion clinica."""
    merged = df[[col_inst, col_clin]].dropna()
    if len(merged) < 5:
        return dict(rho=np.nan, p=np.nan, n=len(merged), sig='', note='n insuficiente')
    if merged[col_clin].std() == 0:
        return dict(rho=np.nan, p=np.nan, n=len(merged), sig='',
                    note='Puntuacion clinica constante - Spearman no calculable')
    rho, p = spearmanr(merged[col_inst], merged[col_clin])
    return dict(
        rho=float(rho), p=float(p), n=int(len(merged)),
        sig=('***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'),
        note='',
    )


# -- Analisis de 3 grupos: Kruskal-Wallis + post-hoc --------------------------
# Grupos: control (h01-h20), ciego (b01-b10), ojos_cerrados (b11-b20).
# El grupo de 20 "invidentes" del analisis de 2 grupos se desglosa aqui en
# ciegos reales (privacion visual permanente) y videntes con ojos cerrados
# (privacion visual simulada).

def asignar_grupo3(subject):
    s = str(subject).lower()
    if s.startswith('h'):
        return 'control'
    try:
        n = int(s[1:])
    except ValueError:
        return ''
    return 'ciego' if 1 <= n <= 10 else 'ojos_cerrados'


GRUPOS3 = ['ciego', 'ojos_cerrados', 'control']


def kw_test(df, col, group_col='group3'):
    """
    Kruskal-Wallis entre los 3 grupos. Devuelve medianas, n, H, p y el tamano
    del efecto epsilon^2 = H/(N-1).
    """
    samples = [df.loc[df[group_col] == g, col].dropna().values for g in GRUPOS3]
    ns = [int(len(s)) for s in samples]
    out = dict(n_ciego=ns[0], n_ojos=ns[1], n_control=ns[2],
               med_ciego=np.nan, med_ojos=np.nan, med_control=np.nan,
               H=np.nan, p=np.nan, eps2=np.nan, sig='')
    if any(n < 2 for n in ns):
        return out
    try:
        H, p = kruskal(*samples)
    except ValueError:   # todos los valores identicos (p.ej. clinica constante)
        return out
    N = sum(ns)
    out.update(med_ciego=float(np.median(samples[0])),
               med_ojos=float(np.median(samples[1])),
               med_control=float(np.median(samples[2])),
               H=float(H), p=float(p), eps2=float(H / (N - 1)),
               sig=('***' if p < 0.001 else '**' if p < 0.01 else
                    '*' if p < 0.05 else 'ns'))
    return out


def posthoc_pares(df, col, group_col='group3'):
    """
    Post-hoc por pares (Mann-Whitney) con correccion de Bonferroni (3 pares).
    Devuelve dict {par: p_ajustado}.
    """
    pares = [('ciego', 'control'), ('ojos_cerrados', 'control'),
             ('ciego', 'ojos_cerrados')]
    res = {}
    for g1, g2 in pares:
        a = df.loc[df[group_col] == g1, col].dropna().values
        b = df.loc[df[group_col] == g2, col].dropna().values
        if len(a) < 2 or len(b) < 2:
            res[f'{g1}_vs_{g2}'] = np.nan
            continue
        _, p = mannwhitneyu(a, b, alternative='two-sided')
        res[f'{g1}_vs_{g2}'] = float(min(p * 3.0, 1.0))
    return res


# -- Main ---------------------------------------------------------------------

def main():
    if not os.path.exists(CLINICAL_PATH):
        print(f"[ERROR] No se encontro {CLINICAL_PATH}")
        print("        Coloca puntuacion_fisioterapia.csv en la carpeta results/")
        return

    df_clin = load_clinical_scores()
    print(f"[OK] Puntuaciones clinicas cargadas: {len(df_clin)} sujetos "
          f"({(df_clin.group == 'blind').sum()} VIP, "
          f"{(df_clin.group == 'healthy').sum()} sanos)")

    if not os.path.exists(RESULTS_PATH):
        print(f"\n[ERROR] No se encontro {RESULTS_PATH}")
        print("        Ejecuta primero los scripts 01-04 de analisis")
        return

    df_g = pd.read_csv(RESULTS_PATH, sep=';', decimal=',')
    print(f"[OK] resultados_globales.csv cargado: {len(df_g)} sujetos")

    # Limpiar columnas previas de este script
    drop = [c for c in df_g.columns if c.startswith('Clin_') or c.startswith('Inst_')]
    df_g = df_g.drop(columns=drop, errors='ignore')

    # Unir clinicas y construir scores
    df_g = df_g.merge(df_clin.drop(columns='group'), on='subject', how='left')
    df_g = build_scores(df_g)

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    df_g.to_csv(RESULTS_PATH, index=False, sep=';', decimal=',')
    print(f"[OK] resultados_globales.csv actualizado")

    n_inst_compl = int(df_g['Inst_Score_Total'].notna().sum())
    n_clin_compl = int(df_g['Clin_Score_Total'].notna().sum())
    print(f"     Casos completos para el Total 0-12: "
          f"{n_inst_compl} instrumental, {n_clin_compl} clinico")

    # (etiqueta, col_clinica, col_instrumental) para Mann-Whitney
    mw_info = [(lab, clin, scaled) for lab, _src, scaled, clin in ITEMS]
    mw_info.append(('Total 0-12', 'Clin_Score_Total', 'Inst_Score_Total'))

    # -- Cabecera Mann-Whitney -------------------------------------------------
    hdr = (f"{'Test':<12} {'Tipo':<14} "
           f"{'n VIP':>6} {'Med VIP':>9} {'IQR VIP':>9}  "
           f"{'n Sano':>6} {'Med Sano':>9} {'IQR Sano':>9}  "
           f"{'U':>7} {'p':>8} {'r':>6}  {'sig':>3}")
    sep = '-' * len(hdr)

    print(f"\n{'='*len(hdr)}")
    print("MANN-WHITNEY U  -  VIP vs Control")
    print(f"{'='*len(hdr)}\n{hdr}\n{sep}")

    stat_rows = []

    def fmt(x, fs='.3f'):
        return format(x, fs) if (x is not None and not np.isnan(x)) else '   -  '

    for label, col_clin, col_inst in mw_info:
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
                f"   [SW VIP p={fmt(sw['sw_p_blind'],'.3f')} "
                f"Sano p={fmt(sw['sw_p_healthy'],'.3f')}]"
            )
            stat_rows.append(dict(Test=label, Tipo=tipo, Columna=col,
                                  **{k: res[k] for k in res if k != 'sig'},
                                  Sig_MW=res['sig'], **sw))

    print(sep)
    print("ns p>=0.05  * p<0.05  ** p<0.01  *** p<0.001  |  "
          "r rank-biserial: |r|>0.3 medio, |r|>0.5 grande")
    print("SW: Shapiro-Wilk (p<0.05 -> rechaza normalidad -> MW justificado)\n")

    # -- Spearman: instrumental vs clinico -------------------------------------
    spearman_pairs = [(lab, scaled, clin) for lab, _src, scaled, clin in ITEMS]
    spearman_pairs.append(('Total 0-12', 'Inst_Score_Total', 'Clin_Score_Total'))

    hdr_sp = f"{'Test':<12} {'n':>5} {'rho':>7} {'p':>8}  {'sig':>3}  {'Nota'}"
    sep_sp = '-' * 60

    print(f"{'='*60}")
    print("SPEARMAN  -  Score instrumentado vs puntuacion clinica")
    print(f"{'='*60}\n{hdr_sp}\n{sep_sp}")

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
    print("rho: correlacion de rangos de Spearman\n")

    # -- Kruskal-Wallis: 3 grupos (ciego / ojos cerrados / control) ------------
    df_g['group3'] = df_g['subject'].apply(asignar_grupo3)

    hdr_kw = (f"{'Test':<12} {'Tipo':<14} "
              f"{'Med Cie':>8} {'Med Ojo':>8} {'Med Con':>8}  "
              f"{'H':>7} {'p':>8} {'eps2':>6} {'sig':>4}   "
              f"{'p Cie-Con':>9} {'p Ojo-Con':>9} {'p Cie-Ojo':>9}")
    sep_kw = '-' * len(hdr_kw)
    print(f"{'='*len(hdr_kw)}")
    print("KRUSKAL-WALLIS  -  3 grupos (Ciegos / Ojos cerrados / Control)")
    print("post-hoc: Mann-Whitney por pares con correccion de Bonferroni")
    print(f"{'='*len(hdr_kw)}\n{hdr_kw}\n{sep_kw}")

    kw_rows = []
    for label, col_clin, col_inst in mw_info:
        for tipo, col in [('Clinico', col_clin), ('Instrumental', col_inst)]:
            if col is None or col not in df_g.columns:
                continue
            kw = kw_test(df_g, col)
            ph = posthoc_pares(df_g, col)
            print(
                f"{label:<12} {tipo:<14} "
                f"{fmt(kw['med_ciego']):>8} {fmt(kw['med_ojos']):>8} {fmt(kw['med_control']):>8}  "
                f"{fmt(kw['H'],'.2f'):>7} {fmt(kw['p'],'.4f'):>8} {fmt(kw['eps2']):>6} {kw['sig']:>4}   "
                f"{fmt(ph['ciego_vs_control'],'.4f'):>9} "
                f"{fmt(ph['ojos_cerrados_vs_control'],'.4f'):>9} "
                f"{fmt(ph['ciego_vs_ojos_cerrados'],'.4f'):>9}"
            )
            kw_rows.append(dict(Test=label, Tipo=tipo, Columna=col,
                                **{k: kw[k] for k in kw if k != 'sig'},
                                Sig_KW=kw['sig'], **ph))

    print(sep_kw)
    print("eps2 = epsilon cuadrado (tamano del efecto, 0-1). "
          "p post-hoc ya ajustados por Bonferroni.\n")

    # -- Guardar tablas estadisticas -------------------------------------------
    pd.DataFrame(stat_rows).to_csv(STATS_PATH, index=False, sep=';', decimal=',')
    sp_path = STATS_PATH.replace('estadistica_MW', 'estadistica_Spearman')
    pd.DataFrame(spearman_rows).to_csv(sp_path, index=False, sep=';', decimal=',')
    kw_path = STATS_PATH.replace('estadistica_MW', 'estadistica_KW')
    pd.DataFrame(kw_rows).to_csv(kw_path, index=False, sep=';', decimal=',')
    print(f"[OK] Tabla MW guardada       -> {STATS_PATH}")
    print(f"[OK] Tabla Spearman guardada -> {sp_path}")
    print(f"[OK] Tabla KW (3 grupos)     -> {kw_path}")

    # -- Resumen por sujeto ----------------------------------------------------
    show = (['subject', 'group'] + [it[2] for it in ITEMS] +
            ['Inst_Score_Total', 'Clin_Score_Total'])
    show = [c for c in show if c in df_g.columns]
    print("\nScores instrumentales escalados (0-2 por sub-ensayo, 0-12 total):")
    print(df_g[show].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
