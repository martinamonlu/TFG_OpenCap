"""
05_control_calidad.py — Control de calidad de los ensayos analizados
=====================================================================
Recorre todos los ensayos de los tests que se analizan (3, 7, 11, 14) y
aplica criterios OBJETIVOS de validez, fijados a priori y aplicados por igual
a ambos grupos. Para cada ensayo emite una de tres decisiones:

    EXCLUIR  — artefacto inequívoco: no hay medida válida.
               (archivo ausente, grabación corrupta, rangos articulares
                clampeados a ±180°, o clip físicamente imposible)
    REVISAR  — corto pero fisiológicamente plausible: podría ser un fallo de
               grabación O un mal rendimiento real. Requiere revisión manual
               en OpenCap antes de decidir.
    OK       — ensayo válido.

IMPORTANTE (rigor estadístico):
    La exclusión se basa SOLO en la calidad de la grabación, nunca en el valor
    del score. Excluir un ensayo por tener un score bajo sesgaría el p-valor.

Genera:
    results/control_calidad.csv  — tabla con todos los ensayos y su decisión.
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import ALL_SUBJECTS, get_mot_path, load_mot, get_trimmed_path

RESULTS_DIR = r'C:\Users\marti\Desktop\TFG\results'
QC_PATH     = os.path.join(RESULTS_DIR, 'control_calidad.csv')

# Tests que entran en el análisis estadístico
ANALYZED_TESTS = ['test3a', 'test3b', 'test7', 'test11', 'test14-1', 'test14-2']

# ── Umbrales de duración (s) ──────────────────────────────────────────────────
# DUR_IMPOSIBLE: por debajo de esto el movimiento del test NO PUEDE ocurrir
#                físicamente → artefacto seguro → EXCLUIR.
# DUR_MINIMA   : por debajo de esto el ensayo es sospechosamente corto pero
#                podría ser rendimiento real → REVISAR a ojo.
#
# Justificación:
#   test3 (apoyo monopodal): mantenerse <2 s no permite estimar el sway de forma
#       fiable; <8 s es corto para la consigna del test pero puede ser mal
#       equilibrio real.
#   test7 (Romberg/superficie inestable): la consigna pide ~20-30 s; <2 s es
#       imposible, <15 s es corto.
#   test11 (marcha con giros de cabeza): un pasillo de marcha dura varios s;
#       <2 s no contiene marcha, <5 s es corto.
#   test14 (TUG): levantarse+marcha+giro+vuelta+sentarse no cabe en <3 s; <4 s
#       es sospechoso.
DUR_IMPOSIBLE = {'test3a': 2.0, 'test3b': 2.0, 'test7': 2.0,
                 'test11': 2.0, 'test14-1': 3.0, 'test14-2': 3.0}
DUR_MINIMA    = {'test3a': 8.0, 'test3b': 8.0, 'test7': 15.0,
                 'test11': 5.0, 'test14-1': 4.0, 'test14-2': 4.0}

# Columnas articulares que el modelo OpenSim clampa a ±180° cuando la
# reconstrucción falla (rango ≥179° = grabación no válida)
CLAMP_COLS  = ['lumbar_bending', 'lumbar_extension', 'lumbar_rotation']
CLAMP_LIMIT = 179.0


def evaluar_ensayo(subject, test):
    """
    Evalúa un único ensayo. Devuelve un dict con la decisión y el motivo.
    """
    row = dict(subject=subject, test=test, n_muestras=np.nan,
               duracion_s=np.nan, rango_max_tronco_deg=np.nan,
               decision='', motivo='')

    # 1) ¿Existe el archivo?
    try:
        get_mot_path(subject, test)
    except FileNotFoundError:
        row['decision'] = 'EXCLUIR'
        row['motivo']   = 'archivo ausente (no grabado/exportado)'
        return row

    # 2) ¿Se generó el recortado? (si no, trim_all dio ERROR de carga)
    try:
        path = get_trimmed_path(subject, test)
    except FileNotFoundError:
        row['decision'] = 'EXCLUIR'
        row['motivo']   = 'grabación corrupta (fallo de carga en trim_all)'
        return row

    df = load_mot(path)
    n  = len(df)
    dur = float(df['time'].iloc[-1] - df['time'].iloc[0]) if n > 1 else 0.0
    row['n_muestras'] = n
    row['duracion_s'] = round(dur, 2)

    # 3) ¿Rangos articulares clampeados a ±180°?
    rng_max = 0.0
    for col in CLAMP_COLS:
        if col in df.columns:
            rng = float(df[col].max() - df[col].min())
            rng_max = max(rng_max, rng)
    row['rango_max_tronco_deg'] = round(rng_max, 1)
    if rng_max >= CLAMP_LIMIT:
        row['decision'] = 'EXCLUIR'
        row['motivo']   = f'rangos de tronco clampeados a ±180° (máx={rng_max:.0f}°)'
        return row

    # 4) Duración físicamente imposible
    if dur < DUR_IMPOSIBLE.get(test, 0.0):
        row['decision'] = 'EXCLUIR'
        row['motivo']   = (f'clip físicamente imposible ({dur:.1f}s < '
                           f'{DUR_IMPOSIBLE[test]:.0f}s)')
        return row

    # 5) Corto pero plausible → revisar a ojo
    if dur < DUR_MINIMA.get(test, 0.0):
        row['decision'] = 'REVISAR'
        row['motivo']   = (f'clip corto ({dur:.1f}s < {DUR_MINIMA[test]:.0f}s) — '
                           f'¿fallo de seguimiento o rendimiento real?')
        return row

    row['decision'] = 'OK'
    return row


def main():
    rows = [evaluar_ensayo(s, t) for s in ALL_SUBJECTS for t in ANALYZED_TESTS]
    df = pd.DataFrame(rows)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(QC_PATH, index=False, sep=';', decimal=',')

    # ── Resumen por pantalla ──────────────────────────────────────────────────
    print('═' * 78)
    print('CONTROL DE CALIDAD — ensayos de los tests 3, 7, 11 y 14')
    print('═' * 78)

    counts = df['decision'].value_counts()
    total  = len(df)
    print(f'Total ensayos evaluados: {total}')
    for dec in ['OK', 'REVISAR', 'EXCLUIR']:
        print(f'  {dec:8}: {int(counts.get(dec, 0))}')

    for dec, titulo in [('EXCLUIR', 'EXCLUIR — artefacto, sin medida válida'),
                        ('REVISAR', 'REVISAR — corto, comprobar en OpenCap')]:
        sub = df[df.decision == dec].sort_values(['test', 'subject'])
        if len(sub) == 0:
            continue
        print(f'\n{"─"*78}\n{titulo}\n{"─"*78}')
        print(f'{"sujeto":8}{"test":11}{"dur(s)":9}{"motivo"}')
        for _, r in sub.iterrows():
            ds = f'{r.duracion_s:.1f}' if not np.isnan(r.duracion_s) else '-'
            print(f'{r.subject:8}{r.test:11}{ds:9}{r.motivo}')

    print(f'\n{"═"*78}')
    print(f'✓ Tabla completa guardada → {QC_PATH}')
    print('  Los EXCLUIR deben ponerse a NaN antes de la estadística.')
    print('  Los REVISAR requieren tu inspección visual en OpenCap.')


if __name__ == '__main__':
    main()
