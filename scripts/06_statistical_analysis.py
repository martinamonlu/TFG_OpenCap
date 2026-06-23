"""
06_statistical_analysis.py - Estadistica con arbol de decision (2 y 3 grupos)
=============================================================================
Requiere en results/:
  - resultados_globales.csv     (generado por los scripts 01-05)
  - puntuacion_fisioterapia.csv (puntuaciones clinicas del fisioterapeuta)

Cada test del Mini-BESTest se puntua 0-2. Se tratan los 6 SUB-ENSAYOS
instrumentados por separado (Test 3a, 3b, 7, 11, 14-1, 14-2), escalados x2, y
el Score Total 0-12 (suma de los 6, casos completos).

ARBOL DE DECISION (para variables CONTINUAS = scores instrumentales):
  1. Shapiro-Wilk por grupo (normalidad) + Levene (homocedasticidad).
  2. Si TODOS los grupos son normales Y hay igualdad de varianzas -> ruta
     PARAMETRICA; si no -> ruta NO PARAMETRICA.
  3. Comparacion:
       2 grupos: t-Student (parametrica) o Mann-Whitney U (no parametrica).
       3 grupos: ANOVA + post-hoc Tukey (parametrica), o Kruskal-Wallis +
                 post-hoc Dunn con Bonferroni (no parametrica).
  4. Tamano del efecto (independiente de la muestra):
       Cohen d (t) | Rosenthal r (MW) | eta2 (ANOVA) | epsilon2 (KW).

Las puntuaciones CLINICAS son ORDINALES (0-2): se comparan SIEMPRE por la ruta
no parametrica (la t/ANOVA no son validas en escala ordinal).

Validacion frente a la clinica: correlacion de Spearman (instrumental vs clinico).
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import (mannwhitneyu, spearmanr, shapiro, kruskal,
                         levene, ttest_ind, f_oneway, tukey_hsd)
import scikit_posthocs as sp

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# -- Rutas --------------------------------------------------------------------

RESULTS_PATH  = r'C:\Users\marti\Desktop\TFG\results\resultados_globales.csv'
CLINICAL_PATH = r'C:\Users\marti\Desktop\TFG\results\puntuacion_fisioterapia.csv'
STATS_DIR     = r'C:\Users\marti\Desktop\TFG\results'

ALPHA = 0.05

# Los 6 sub-ensayos: (etiqueta, col_origen, col_escalada, col_clinica)
ITEMS = [
    ('Test 3a',   'T3A_Score',   'Inst_Test3a_2',   'Clin_Test3a'),
    ('Test 3b',   'T3B_Score',   'Inst_Test3b_2',   'Clin_Test3b'),
    ('Test 7',    'T7_Score',    'Inst_Test7_2',    'Clin_Test7'),
    ('Test 11',   'T11_Score',   'Inst_Test11_2',   'Clin_Test11'),
    ('Test 14-1', 'T14_1_Score', 'Inst_Test14_1_2', 'Clin_Test14_1'),
    ('Test 14-2', 'T14_2_Score', 'Inst_Test14_2_2', 'Clin_Test14_2'),
]

# Grupos de 3: control (h01-h20), disc_visual (b01-b11), ojos_cerrados (b12-b20)
GRUPOS3 = ['disc_visual', 'ojos_cerrados', 'control']
PARES3  = [('disc_visual', 'control'), ('ojos_cerrados', 'control'),
           ('disc_visual', 'ojos_cerrados')]


# -- Parsear puntuaciones clinicas --------------------------------------------

def load_clinical_scores():
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
        rows.append(dict(
            subject=subject,
            group='blind' if subject.startswith('b') else 'healthy',
            Clin_Test3a   = to_float(row.iloc[COL['t3a']]),
            Clin_Test3b   = to_float(row.iloc[COL['t3b']]),
            Clin_Test7    = to_float(row.iloc[COL['t7']]),
            Clin_Test11   = to_float(row.iloc[COL['t11']]),
            Clin_Test14_1 = to_float(row.iloc[COL['t14_1']]),
            Clin_Test14_2 = to_float(row.iloc[COL['t14_2']]),
        ))
    return pd.DataFrame(rows)


def asignar_grupo3(subject):
    s = str(subject).lower()
    if s.startswith('h'):
        return 'control'
    try:    n = int(s[1:])
    except ValueError: return ''
    return 'disc_visual' if 1 <= n <= 11 else 'ojos_cerrados'


# -- Escalar instrumentales x2 y totales 0-12 ---------------------------------

def build_scores(df):
    inst_scaled = []
    for _, src, scaled, _clin in ITEMS:
        df[scaled] = df[src] * 2.0 if src in df.columns else np.nan
        inst_scaled.append(scaled)
    clin_cols = [it[3] for it in ITEMS]

    def total_completo(cols):
        presentes = [c for c in cols if c in df.columns]
        if len(presentes) < len(cols):
            return pd.Series(np.nan, index=df.index)
        completo = df[presentes].notna().all(axis=1)
        return pd.Series(np.where(completo, df[presentes].sum(axis=1), np.nan),
                         index=df.index)

    df['Inst_Score_Total'] = total_completo(inst_scaled)
    df['Clin_Score_Total'] = total_completo(clin_cols)
    return df


# -- Tamanos del efecto -------------------------------------------------------

def cohen_d(a, b):
    n1, n2 = len(a), len(b)
    sp_ = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1))
                  / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / sp_) if sp_ > 0 else np.nan


def rosenthal_r(U, n1, n2):
    """r = Z/sqrt(N), con Z del estadistico U de Mann-Whitney."""
    N = n1 + n2
    mu = n1 * n2 / 2.0
    sigma = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if sigma == 0:
        return np.nan
    z = (U - mu) / sigma
    return float(z / np.sqrt(N))


def eta2_anova(F, k, N):
    dfb, dfw = k - 1, N - k
    den = F * dfb + dfw
    return float(F * dfb / den) if den > 0 else np.nan


# -- Validacion de supuestos y decision de ruta -------------------------------

def shapiro_p(x):
    x = np.asarray(x, dtype=float)
    if 3 <= len(x) <= 5000 and np.std(x) > 0:
        return float(shapiro(x).pvalue)
    return np.nan


def decidir_ruta(samples, ordinal):
    """
    Devuelve (ruta, lista_p_shapiro, p_levene).
    Ordinal -> siempre 'no_parametrica'. Continua -> parametrica solo si todos
    los grupos son normales (Shapiro p>0.05) y hay homocedasticidad (Levene p>0.05).
    """
    sw = [shapiro_p(s) for s in samples]
    if ordinal:
        return 'no_parametrica', sw, np.nan
    try:
        p_lev = float(levene(*samples).pvalue)
    except Exception:
        p_lev = np.nan
    normal  = all((not np.isnan(p)) and p > ALPHA for p in sw)
    homoced = (not np.isnan(p_lev)) and p_lev > ALPHA
    ruta = 'parametrica' if (normal and homoced) else 'no_parametrica'
    return ruta, sw, p_lev


def label_r(r):
    a = abs(r)
    return 'grande' if a >= 0.5 else 'medio' if a >= 0.3 else 'pequeno'


# -- Comparacion de 2 grupos --------------------------------------------------

def comparar_2(a, b, ordinal=False):
    a = np.asarray(a, float); b = np.asarray(b, float)
    out = dict(n1=len(a), n2=len(b), med1=np.nan, med2=np.nan,
               sw1=np.nan, sw2=np.nan, p_levene=np.nan,
               ruta='', test='', stat=np.nan, p=np.nan,
               effect=np.nan, effect_name='', sig='')
    if len(a) < 2 or len(b) < 2 or np.std(np.concatenate([a, b])) == 0:
        return out
    ruta, sw, p_lev = decidir_ruta([a, b], ordinal)
    out.update(med1=float(np.median(a)), med2=float(np.median(b)),
               sw1=sw[0], sw2=sw[1], p_levene=p_lev, ruta=ruta)
    if ruta == 'parametrica':
        t, p = ttest_ind(a, b, equal_var=True)
        out.update(test='t-Student', stat=float(t), p=float(p),
                   effect=cohen_d(a, b), effect_name='Cohen d')
    else:
        U, p = mannwhitneyu(a, b, alternative='two-sided')
        out.update(test='Mann-Whitney U', stat=float(U), p=float(p),
                   effect=rosenthal_r(U, len(a), len(b)), effect_name='Rosenthal r')
    out['sig'] = '*' if out['p'] < ALPHA else 'ns'
    return out


# -- Comparacion de 3 grupos --------------------------------------------------

def comparar_3(samples, ordinal=False):
    out = dict(ruta='', test='', stat=np.nan, p=np.nan, effect=np.nan,
               effect_name='', sw=[np.nan] * 3, p_levene=np.nan, sig='',
               posthoc={f'{g1}_vs_{g2}': np.nan for g1, g2 in PARES3})
    if any(len(s) < 2 for s in samples):
        return out
    todos = np.concatenate([np.asarray(s, float) for s in samples])
    if np.std(todos) == 0:        # constante (p.ej. clinica Test 7) -> no comparable
        return out
    ruta, sw, p_lev = decidir_ruta(samples, ordinal)
    out.update(ruta=ruta, sw=sw, p_levene=p_lev)
    N = sum(len(s) for s in samples); k = len(samples)
    idx = {g: i for i, g in enumerate(GRUPOS3)}

    if ruta == 'parametrica':
        F, p = f_oneway(*samples)
        out.update(test='ANOVA', stat=float(F), p=float(p),
                   effect=eta2_anova(F, k, N), effect_name='eta2')
        try:
            th = tukey_hsd(*samples)
            for g1, g2 in PARES3:
                out['posthoc'][f'{g1}_vs_{g2}'] = float(th.pvalue[idx[g1], idx[g2]])
        except Exception:
            pass
    else:
        H, p = kruskal(*samples)
        out.update(test='Kruskal-Wallis', stat=float(H), p=float(p),
                   effect=float(H / (N - 1)), effect_name='epsilon2')
        try:
            dunn = sp.posthoc_dunn(list(samples), p_adjust='bonferroni')
            di = {g: i + 1 for i, g in enumerate(GRUPOS3)}   # dunn indexa 1..k
            for g1, g2 in PARES3:
                out['posthoc'][f'{g1}_vs_{g2}'] = float(dunn.loc[di[g1], di[g2]])
        except Exception:
            pass
    out['sig'] = '*' if out['p'] < ALPHA else 'ns'
    return out


# -- Spearman -----------------------------------------------------------------

def spearman_test(df, col_inst, col_clin):
    m = df[[col_inst, col_clin]].dropna()
    if len(m) < 5:
        return dict(rho=np.nan, p=np.nan, n=len(m), sig='', note='n insuficiente')
    if m[col_clin].std() == 0:
        return dict(rho=np.nan, p=np.nan, n=len(m), sig='',
                    note='Puntuacion clinica constante - Spearman no calculable')
    rho, p = spearmanr(m[col_inst], m[col_clin])
    return dict(rho=float(rho), p=float(p), n=int(len(m)),
                sig='*' if p < ALPHA else 'ns', note='')


# -- Main ---------------------------------------------------------------------

def fmt(x, fs='.3f'):
    return format(x, fs) if (x is not None and not (isinstance(x, float) and np.isnan(x))) else '  -  '


def main():
    if not (os.path.exists(CLINICAL_PATH) and os.path.exists(RESULTS_PATH)):
        print("[ERROR] Faltan resultados_globales.csv o puntuacion_fisioterapia.csv")
        return

    df_clin = load_clinical_scores()
    df_g = pd.read_csv(RESULTS_PATH, sep=';', decimal=',')
    print(f"[OK] {len(df_g)} sujetos | clinicas: {len(df_clin)}")

    df_g = df_g.drop(columns=[c for c in df_g.columns
                              if c.startswith('Clin_') or c.startswith('Inst_')],
                     errors='ignore')
    df_g = df_g.merge(df_clin.drop(columns='group'), on='subject', how='left')
    df_g = build_scores(df_g)
    df_g['group3'] = df_g['subject'].apply(asignar_grupo3)
    df_g.to_csv(RESULTS_PATH, index=False, sep=';', decimal=',')
    print(f"[OK] resultados_globales.csv actualizado. Casos completos Total 0-12: "
          f"{int(df_g['Inst_Score_Total'].notna().sum())} inst, "
          f"{int(df_g['Clin_Score_Total'].notna().sum())} clin")

    info = [(lab, clin, inst) for lab, _s, inst, clin in ITEMS]
    info.append(('Total 0-12', 'Clin_Score_Total', 'Inst_Score_Total'))

    # ===================== 2 GRUPOS (invidentes vs control) ==================
    print(f"\n{'='*108}")
    print("2 GRUPOS: invidentes vs control  (arbol de decision; clinico = ordinal -> no parametrico)")
    print('='*108)
    hdr = (f"{'Test':<11}{'Tipo':<13}{'n1/n2':>7} {'SW1':>6} {'SW2':>6} {'Lev':>6} "
           f"{'Ruta':>14} {'Test':<15}{'p':>8} {'efecto':>16} {'sig':>4}")
    print(hdr + "\n" + '-'*len(hdr))
    rows2 = []
    for lab, col_clin, col_inst in info:
        for tipo, col, ordinal in [('Clinico', col_clin, True),
                                    ('Instrumental', col_inst, False)]:
            if col not in df_g.columns:
                continue
            a = df_g.loc[df_g.group == 'blind', col].dropna().values
            b = df_g.loc[df_g.group == 'healthy', col].dropna().values
            r = comparar_2(a, b, ordinal=ordinal)
            ef = (f"{r['effect_name']}={fmt(r['effect'],'.2f')}"
                  if r['effect_name'] else '-')
            print(f"{lab:<11}{tipo:<13}{r['n1']:>3}/{r['n2']:<3} "
                  f"{fmt(r['sw1'],'.3f'):>6} {fmt(r['sw2'],'.3f'):>6} "
                  f"{fmt(r['p_levene'],'.3f'):>6} {r['ruta']:>14} {r['test']:<15}"
                  f"{fmt(r['p'],'.4f'):>8} {ef:>16} {r['sig']:>4}")
            rows2.append(dict(Test=lab, Tipo=tipo, **r))

    # ===================== 3 GRUPOS ==========================================
    print(f"\n{'='*120}")
    print("3 GRUPOS: discapacidad visual / ojos cerrados / control  (post-hoc Tukey o Dunn-Bonferroni)")
    print('='*120)
    hdr3 = (f"{'Test':<11}{'Tipo':<13}{'Ruta':>14} {'Test':<15}{'p':>8} "
            f"{'efecto':>14} {'sig':>4}  {'p DV-Con':>9}{'p Ojo-Con':>10}{'p DV-Ojo':>10}")
    print(hdr3 + "\n" + '-'*len(hdr3))
    rows3 = []
    for lab, col_clin, col_inst in info:
        for tipo, col, ordinal in [('Clinico', col_clin, True),
                                    ('Instrumental', col_inst, False)]:
            if col not in df_g.columns:
                continue
            samples = [df_g.loc[df_g.group3 == g, col].dropna().values for g in GRUPOS3]
            r = comparar_3(samples, ordinal=ordinal)
            ef = (f"{r['effect_name']}={fmt(r['effect'],'.2f')}" if r['effect_name'] else '-')
            ph = r['posthoc']
            print(f"{lab:<11}{tipo:<13}{r['ruta']:>14} {r['test']:<15}"
                  f"{fmt(r['p'],'.4f'):>8} {ef:>14} {r['sig']:>4}  "
                  f"{fmt(ph['disc_visual_vs_control'],'.4f'):>9}"
                  f"{fmt(ph['ojos_cerrados_vs_control'],'.4f'):>10}"
                  f"{fmt(ph['disc_visual_vs_ojos_cerrados'],'.4f'):>10}")
            fila = dict(Test=lab, Tipo=tipo, ruta=r['ruta'], test=r['test'],
                        stat=r['stat'], p=r['p'], effect=r['effect'],
                        effect_name=r['effect_name'], p_levene=r['p_levene'],
                        sig=r['sig'], **{f'posthoc_{k}': v for k, v in ph.items()})
            rows3.append(fila)

    # ===================== SPEARMAN ==========================================
    print(f"\n{'='*60}")
    print("SPEARMAN: score instrumentado vs puntuacion clinica")
    print('='*60)
    print(f"{'Test':<12}{'n':>5}{'rho':>8}{'p':>9}{'sig':>5}  Nota")
    pares_sp = [(lab, inst, clin) for lab, _s, inst, clin in ITEMS]
    pares_sp.append(('Total 0-12', 'Inst_Score_Total', 'Clin_Score_Total'))
    rows_sp = []
    for lab, ci, cc in pares_sp:
        if ci in df_g.columns and cc in df_g.columns:
            spr = spearman_test(df_g, ci, cc)
            print(f"{lab:<12}{spr['n']:>5}{fmt(spr['rho']):>8}{fmt(spr['p'],'.4f'):>9}"
                  f"{spr['sig']:>5}  {spr['note']}")
            rows_sp.append(dict(Test=lab, **spr))

    # ===================== GUARDAR ===========================================
    pd.DataFrame(rows2).to_csv(os.path.join(STATS_DIR, 'estadistica_2grupos.csv'),
                               index=False, sep=';', decimal=',')
    pd.DataFrame(rows3).to_csv(os.path.join(STATS_DIR, 'estadistica_3grupos.csv'),
                               index=False, sep=';', decimal=',')
    pd.DataFrame(rows_sp).to_csv(os.path.join(STATS_DIR, 'estadistica_Spearman.csv'),
                                 index=False, sep=';', decimal=',')
    print(f"\n[OK] Tablas -> estadistica_2grupos.csv, estadistica_3grupos.csv, "
          f"estadistica_Spearman.csv")
    print("Effect size: Cohen d (t) | Rosenthal r (MW) | eta2 (ANOVA) | epsilon2 (KW).")


if __name__ == '__main__':
    main()
