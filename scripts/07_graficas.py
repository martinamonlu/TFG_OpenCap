"""
07_graficas.py - Representaciones graficas de la estadistica (MW y Spearman)
=============================================================================
Genera figuras en results/figuras/ a partir de:
  - resultados_globales.csv      (valores por sujeto)
  - estadistica_MW.csv           (p y r de Mann-Whitney, ya calculados en 06)
  - estadistica_Spearman.csv     (rho, p, n, ya calculados en 06)
  - estadistica_KW.csv           (p y eps2 de Kruskal-Wallis, ya calculados en 06)

Figuras:
  1. boxplots_subtests.png  - diagramas de caja VIP vs Control de los 6
       sub-ensayos instrumentales (escala 0-2), con puntos individuales y el
       p-valor exacto + tamano del efecto r sobre cada par.
  2. boxplot_total.png      - diagramas de caja del Score Total 0-12
       (instrumental y clinico), VIP vs Control.
  3. spearman_scatter.png   - dispersion instrumental vs clinico por test,
       con linea de tendencia indicativa y rho + p anotados.

Rigor: se anota el p-valor EXACTO y el tamano del efecto, no asteriscos.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RESULTS_DIR = r'C:\Users\marti\Desktop\TFG\results'
FIG_DIR     = os.path.join(RESULTS_DIR, 'figuras')

# Colores Okabe-Ito (seguros para daltonismo e impresion)
COL_VIP    = '#E69F00'   # naranja  (Invidentes, 2 grupos)
COL_CON    = '#0072B2'   # azul     (Control)
COL_CIEGO  = '#E69F00'   # naranja  (Discapacidad visual, 3 grupos)
COL_OJOS   = '#9467BD'   # morado   (Ojos cerrados, 3 grupos)

# (etiqueta, col_instrumental, col_clinica)
ITEMS = [
    ('Test 3a',   'Inst_Test3a_2',   'Clin_Test3a'),
    ('Test 3b',   'Inst_Test3b_2',   'Clin_Test3b'),
    ('Test 7',    'Inst_Test7_2',    'Clin_Test7'),
    ('Test 11',   'Inst_Test11_2',   'Clin_Test11'),
    ('Test 14-1', 'Inst_Test14_1_2', 'Clin_Test14_1'),
    ('Test 14-2', 'Inst_Test14_2_2', 'Clin_Test14_2'),
]


def fmt_p(p):
    if p is None or np.isnan(p):
        return 'p = n/d'
    return 'p < 0.001' if p < 0.001 else f'p = {p:.3f}'


def cargar():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'resultados_globales.csv'),
                     sep=';', decimal=',')
    mw = pd.read_csv(os.path.join(RESULTS_DIR, 'estadistica_MW.csv'),
                     sep=';', decimal=',')
    sp = pd.read_csv(os.path.join(RESULTS_DIR, 'estadistica_Spearman.csv'),
                     sep=';', decimal=',')
    kw = pd.read_csv(os.path.join(RESULTS_DIR, 'estadistica_KW.csv'),
                     sep=';', decimal=',')
    return df, mw, sp, kw


def get_mw(mw, label):
    """Devuelve (p, r) del Mann-Whitney instrumental de un test."""
    row = mw[(mw['Test'] == label) & (mw['Tipo'] == 'Instrumental')]
    if len(row):
        return float(row['p'].iloc[0]), float(row['r'].iloc[0])
    return np.nan, np.nan


def get_sp(sp, label):
    """Devuelve (rho, p, n) de Spearman de un test."""
    row = sp[sp['Test'] == label]
    if len(row):
        return (float(row['rho'].iloc[0]), float(row['p'].iloc[0]),
                int(row['n'].iloc[0]))
    return np.nan, np.nan, 0


def get_kw(kw, label):
    """Devuelve (p, eps2) del Kruskal-Wallis instrumental de un test."""
    row = kw[(kw['Test'] == label) & (kw['Tipo'] == 'Instrumental')]
    if len(row):
        return float(row['p'].iloc[0]), float(row['eps2'].iloc[0])
    return np.nan, np.nan


def grupo3(subject):
    s = str(subject).lower()
    if s.startswith('h'):
        return 'control'
    n = int(s[1:])
    return 'disc_visual' if 1 <= n <= 11 else 'ojos_cerrados'


def _jitter(n, w=0.12):
    return np.random.uniform(-w, w, n)


# ---------------------------------------------------------------------------
# Figura 1: boxplots de los 6 sub-ensayos instrumentales
# ---------------------------------------------------------------------------

def fig_boxplots_subtests(df, mw):
    np.random.seed(0)
    fig, ax = plt.subplots(figsize=(11, 5.5))
    paso = 2.5

    for i, (lab, col_inst, _clin) in enumerate(ITEMS):
        a = df.loc[df.group == 'blind',   col_inst].dropna().values
        b = df.loc[df.group == 'healthy', col_inst].dropna().values
        x_vip, x_con = i * paso + 0.7, i * paso + 1.5

        for x, data, color in [(x_vip, a, COL_VIP), (x_con, b, COL_CON)]:
            bp = ax.boxplot([data], positions=[x], widths=0.6,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='black', linewidth=1.6))
            bp['boxes'][0].set(facecolor=color, alpha=0.45)
            ax.scatter(x + _jitter(len(data)), data, s=18, color=color,
                       edgecolor='black', linewidth=0.3, alpha=0.8, zorder=3)

        # Anotacion: p exacto + r encima del par
        p, r = get_mw(mw, lab)
        ymax = max(a.max() if len(a) else 0, b.max() if len(b) else 0)
        ytxt = ymax + 0.18
        ax.plot([x_vip, x_con], [ytxt - 0.04, ytxt - 0.04], color='gray', lw=0.8)
        ax.text((x_vip + x_con) / 2, ytxt, f'{fmt_p(p)}\nr = {r:.2f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks([i * paso + 1.1 for i in range(len(ITEMS))])
    ax.set_xticklabels([it[0] for it in ITEMS])
    ax.set_ylabel('Puntuación instrumental (escala 0–2)')
    ax.set_ylim(-0.1, 2.6)
    ax.set_title('Comparación Invidentes vs. Control por sub-ensayo (Mann-Whitney U)')
    ax.grid(axis='y', alpha=0.25)

    handles = [plt.Line2D([0], [0], marker='s', color='w', label='Invidentes',
                          markerfacecolor=COL_VIP, markersize=10),
               plt.Line2D([0], [0], marker='s', color='w', label='Control',
                          markerfacecolor=COL_CON, markersize=10)]
    ax.legend(handles=handles, loc='upper right', framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'boxplots_subtests.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figura 2: boxplot del Score Total 0-12 (instrumental y clinico)
# ---------------------------------------------------------------------------

def fig_boxplot_total(df, mw):
    np.random.seed(1)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    bloques = [('Total instrumental', 'Inst_Score_Total'),
               ('Total clínico',      'Clin_Score_Total')]

    for i, (lab, col) in enumerate(bloques):
        a = df.loc[df.group == 'blind',   col].dropna().values
        b = df.loc[df.group == 'healthy', col].dropna().values
        x_vip, x_con = i * 2.5 + 0.7, i * 2.5 + 1.5

        for x, data, color in [(x_vip, a, COL_VIP), (x_con, b, COL_CON)]:
            bp = ax.boxplot([data], positions=[x], widths=0.6,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='black', linewidth=1.6))
            bp['boxes'][0].set(facecolor=color, alpha=0.45)
            ax.scatter(x + _jitter(len(data)), data, s=20, color=color,
                       edgecolor='black', linewidth=0.3, alpha=0.8, zorder=3)

        label_mw = 'Total 0-12'
        tipo = 'Instrumental' if 'instrumental' in lab else 'Clinico'
        row = mw[(mw['Test'] == label_mw) & (mw['Tipo'] == tipo)]
        if len(row):
            p, r = float(row['p'].iloc[0]), float(row['r'].iloc[0])
            ymax = max(a.max() if len(a) else 0, b.max() if len(b) else 0)
            ax.plot([x_vip, x_con], [ymax + 0.5, ymax + 0.5], color='gray', lw=0.8)
            ax.text((x_vip + x_con) / 2, ymax + 0.7, f'{fmt_p(p)}\nr = {r:.2f}',
                    ha='center', va='bottom', fontsize=9)

    ax.set_xticks([1.1, 3.6])
    ax.set_xticklabels([b[0] for b in bloques])
    ax.set_ylabel('Puntuación Total (escala 0–12)')
    ax.set_ylim(0, 14.5)
    ax.set_title('Score Total Invidentes vs. Control (Mann-Whitney U)', pad=16)
    ax.grid(axis='y', alpha=0.25)
    handles = [plt.Line2D([0], [0], marker='s', color='w', label='Invidentes',
                          markerfacecolor=COL_VIP, markersize=10),
               plt.Line2D([0], [0], marker='s', color='w', label='Control',
                          markerfacecolor=COL_CON, markersize=10)]
    ax.legend(handles=handles, loc='lower right', framealpha=0.9)

    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'boxplot_total.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figura 3: dispersion Spearman (instrumental vs clinico)
# ---------------------------------------------------------------------------

def fig_spearman(df, sp):
    np.random.seed(2)
    # Test 7 se omite: la puntuacion clinica es constante (todos 2) y Spearman
    # no es calculable.
    paneles = [it for it in ITEMS if it[0] != 'Test 7']
    paneles.append(('Total 0-12', 'Inst_Score_Total', 'Clin_Score_Total'))

    ncol = 3
    nrow = int(np.ceil(len(paneles) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 4 * nrow))
    axes = axes.flatten()

    for k, (lab, col_inst, col_clin) in enumerate(paneles):
        ax = axes[k]
        sub = df[[col_inst, col_clin, 'group']].dropna()
        for grp, color, name in [('blind', COL_VIP, 'VIP'),
                                 ('healthy', COL_CON, 'Control')]:
            s = sub[sub.group == grp]
            jit = _jitter(len(s), 0.06) if lab != 'Total 0-12' else 0.0
            ax.scatter(s[col_inst], s[col_clin] + jit, s=26, color=color,
                       edgecolor='black', linewidth=0.3, alpha=0.8, label=name)

        # Linea de tendencia indicativa (solo apoyo visual; Spearman es de rangos)
        if len(sub) >= 2 and sub[col_inst].std() > 0:
            m, c = np.polyfit(sub[col_inst], sub[col_clin], 1)
            xs = np.linspace(sub[col_inst].min(), sub[col_inst].max(), 50)
            ax.plot(xs, m * xs + c, color='gray', lw=1.2, ls='--')

        rho, p, n = get_sp(sp, lab)
        ax.text(0.04, 0.96, f'rho = {rho:.2f}\n{fmt_p(p)}\nn = {n}',
                transform=ax.transAxes, ha='left', va='top', fontsize=9,
                bbox=dict(boxstyle='round', fc='white', ec='gray', alpha=0.8))

        ax.set_title(lab)
        ax.set_xlabel('Instrumental')
        ax.set_ylabel('Clínico')
        ax.grid(alpha=0.25)

    for j in range(len(paneles), len(axes)):
        axes[j].axis('off')
    axes[0].legend(loc='lower right', fontsize=8, framealpha=0.9)

    fig.suptitle('Correlación de Spearman: puntuación instrumental vs. clínica',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(FIG_DIR, 'spearman_scatter.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


# ---------------------------------------------------------------------------
# Figura 4: boxplots de 3 grupos (Ciegos / Ojos cerrados / Control)
# ---------------------------------------------------------------------------

G3 = [('disc_visual', COL_CIEGO, 'Discapacidad visual'),
      ('ojos_cerrados', COL_OJOS, 'Ojos cerrados'),
      ('control', COL_CON, 'Control')]


def _leyenda_g3(ax, loc='upper right'):
    handles = [plt.Line2D([0], [0], marker='s', color='w', label=nom,
                          markerfacecolor=col, markersize=10)
               for _g, col, nom in G3]
    ax.legend(handles=handles, loc=loc, framealpha=0.9)


def fig_boxplots_3grupos(df, kw):
    np.random.seed(3)
    df = df.copy()
    df['g3'] = df['subject'].apply(grupo3)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    paso = 3.0

    for i, (lab, col_inst, _clin) in enumerate(ITEMS):
        ymax = 0
        for j, (g, color, _nom) in enumerate(G3):
            data = df.loc[df.g3 == g, col_inst].dropna().values
            x = i * paso + 0.6 + j * 0.7
            bp = ax.boxplot([data], positions=[x], widths=0.55,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color='black', linewidth=1.5))
            bp['boxes'][0].set(facecolor=color, alpha=0.45)
            ax.scatter(x + _jitter(len(data), 0.10), data, s=14, color=color,
                       edgecolor='black', linewidth=0.3, alpha=0.8, zorder=3)
            ymax = max(ymax, data.max() if len(data) else 0)

        p, eps2 = get_kw(kw, lab)
        xc = i * paso + 0.6 + 0.7
        ax.text(xc, ymax + 0.16, f'KW {fmt_p(p)}\nε² = {eps2:.2f}',
                ha='center', va='bottom', fontsize=8)

    ax.set_xticks([i * paso + 1.3 for i in range(len(ITEMS))])
    ax.set_xticklabels([it[0] for it in ITEMS])
    ax.set_ylabel('Puntuación instrumental (escala 0–2)')
    ax.set_ylim(-0.1, 2.7)
    ax.set_title('Comparación de 3 grupos por sub-ensayo (Kruskal-Wallis)')
    ax.grid(axis='y', alpha=0.25)
    _leyenda_g3(ax)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'boxplots_subtests_3grupos.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def fig_boxplot_total_3grupos(df, kw):
    np.random.seed(4)
    df = df.copy()
    df['g3'] = df['subject'].apply(grupo3)
    fig, ax = plt.subplots(figsize=(7, 5.5))

    ymax = 0
    for j, (g, color, nom) in enumerate(G3):
        data = df.loc[df.g3 == g, 'Inst_Score_Total'].dropna().values
        x = 1 + j * 0.9
        bp = ax.boxplot([data], positions=[x], widths=0.7, patch_artist=True,
                        showfliers=False, medianprops=dict(color='black', linewidth=1.6))
        bp['boxes'][0].set(facecolor=color, alpha=0.45)
        ax.scatter(x + _jitter(len(data), 0.12), data, s=22, color=color,
                   edgecolor='black', linewidth=0.3, alpha=0.8, zorder=3)
        ymax = max(ymax, data.max() if len(data) else 0)

    p, eps2 = get_kw(kw, 'Total 0-12')
    ax.text(1 + 0.9, ymax + 0.6, f'Kruskal-Wallis {fmt_p(p)}   ε² = {eps2:.2f}',
            ha='center', va='bottom', fontsize=10)
    ax.set_xticks([1 + j * 0.9 for j in range(3)])
    ax.set_xticklabels([nom for _g, _c, nom in G3])
    ax.set_ylabel('Puntuación Total instrumental (escala 0–12)')
    ax.set_ylim(0, 13.5)
    ax.set_title('Score Total: 3 grupos (Kruskal-Wallis)', pad=14)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    out = os.path.join(FIG_DIR, 'boxplot_total_3grupos.png')
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df, mw, sp, kw = cargar()
    salidas = [
        fig_boxplots_subtests(df, mw),
        fig_boxplot_total(df, mw),
        fig_spearman(df, sp),
        fig_boxplots_3grupos(df, kw),
        fig_boxplot_total_3grupos(df, kw),
    ]
    print('Figuras guardadas en', FIG_DIR)
    for p in salidas:
        print('  ', os.path.basename(p))


if __name__ == '__main__':
    main()
