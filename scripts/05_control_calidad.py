"""
05_control_calidad.py - Control de calidad de los ensayos analizados
=====================================================================
Genera y mantiene UN UNICO archivo, results/control_calidad.csv, que sirve a
la vez de informe automatico y de hoja de decisiones manuales.

Columnas AUTOMATICAS (las rellena este script cada vez que se ejecuta):
    senal_auto   - EXCLUIR / REVISAR / OK segun criterios objetivos a priori
    motivo_auto  - razon de la senal
    duracion_s, n_muestras, rango_max_tronco_deg

Columnas MANUALES (las edita el experto a mano; este script NO las pisa):
    decision_manual - 'excluir' | 'recorte' | vacio
    trim_inicio_s, trim_fin_s - ventana de recorte manual (solo si 'recorte')
    motivo_manual   - nota libre

Senales automaticas:
    EXCLUIR - artefacto inequivoco: no hay medida valida (archivo ausente,
              grabacion corrupta, rangos articulares en el limite del modelo,
              o clip fisicamente imposible).
    REVISAR - corto pero plausible: revisar en OpenCap antes de decidir.
    OK      - ensayo valido.

La senal es solo eso, una senal. La decision final la toma el experto en la
columna decision_manual, y es la que usa el pipeline (utils.is_excluded /
get_manual_trim). Re-ejecutar este script refresca las columnas automaticas
pero conserva las manuales.

IMPORTANTE (rigor estadistico): la exclusion se basa SOLO en la calidad de la
grabacion, nunca en el valor del score. Excluir por score bajo sesga el p-valor.
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

# Tests que entran en el analisis estadistico
ANALYZED_TESTS = ['test3a', 'test3b', 'test7', 'test11', 'test14-1', 'test14-2']

# Columnas que edita el experto y que este script NO debe pisar
MANUAL_COLS = ['decision_manual', 'trim_inicio_s', 'trim_fin_s', 'motivo_manual']

# -- Umbrales de duracion (s) -------------------------------------------------
# DUR_IMPOSIBLE: por debajo de esto el movimiento del test NO PUEDE ocurrir
#                fisicamente -> artefacto seguro -> EXCLUIR.
# DUR_MINIMA   : por debajo de esto el ensayo es sospechosamente corto pero
#                podria ser rendimiento real -> REVISAR a ojo.
DUR_IMPOSIBLE = {'test3a': 2.0, 'test3b': 2.0, 'test7': 2.0,
                 'test11': 2.0, 'test14-1': 3.0, 'test14-2': 3.0}
DUR_MINIMA    = {'test3a': 8.0, 'test3b': 8.0, 'test7': 15.0,
                 'test11': 5.0, 'test14-1': 4.0, 'test14-2': 4.0}

# Columnas articulares que el modelo OpenSim lleva a su limite cuando la
# reconstruccion falla (rango >=179 grados = grabacion no valida)
CLAMP_COLS  = ['lumbar_bending', 'lumbar_extension', 'lumbar_rotation']
CLAMP_LIMIT = 179.0


def evaluar_ensayo(subject, test):
    """Evalua un ensayo y devuelve un dict con las columnas AUTOMATICAS."""
    row = dict(subject=subject, test=test, senal_auto='', motivo_auto='',
               duracion_s=np.nan, n_muestras=np.nan, rango_max_tronco_deg=np.nan)

    # 1) Existe el archivo?
    try:
        get_mot_path(subject, test)
    except FileNotFoundError:
        row['senal_auto'] = 'EXCLUIR'
        row['motivo_auto'] = 'archivo ausente (no grabado/exportado)'
        return row

    # 2) Se genero el recortado? (si no, trim_all dio ERROR de carga)
    try:
        path = get_trimmed_path(subject, test)
    except FileNotFoundError:
        row['senal_auto'] = 'EXCLUIR'
        row['motivo_auto'] = 'grabacion corrupta (fallo de carga en trim_all)'
        return row

    df = load_mot(path)
    n  = len(df)
    dur = float(df['time'].iloc[-1] - df['time'].iloc[0]) if n > 1 else 0.0
    row['n_muestras'] = n
    row['duracion_s'] = round(dur, 2)

    # 3) Rangos articulares en el limite del modelo?
    rng_max = 0.0
    for col in CLAMP_COLS:
        if col in df.columns:
            rng_max = max(rng_max, float(df[col].max() - df[col].min()))
    row['rango_max_tronco_deg'] = round(rng_max, 1)
    if rng_max >= CLAMP_LIMIT:
        row['senal_auto'] = 'EXCLUIR'
        row['motivo_auto'] = f'rangos de tronco en limite del modelo (max={rng_max:.0f} grados)'
        return row

    # 4) Duracion fisicamente imposible
    if dur < DUR_IMPOSIBLE.get(test, 0.0):
        row['senal_auto'] = 'EXCLUIR'
        row['motivo_auto'] = f'clip fisicamente imposible ({dur:.1f}s < {DUR_IMPOSIBLE[test]:.0f}s)'
        return row

    # 5) Corto pero plausible -> revisar a ojo
    if dur < DUR_MINIMA.get(test, 0.0):
        row['senal_auto'] = 'REVISAR'
        row['motivo_auto'] = (f'clip corto ({dur:.1f}s < {DUR_MINIMA[test]:.0f}s) - '
                              f'fallo de seguimiento o rendimiento real?')
        return row

    row['senal_auto'] = 'OK'
    return row


def cargar_decisiones_previas():
    """
    Lee las columnas manuales del control_calidad.csv existente para no
    pisarlas. Devuelve {(subject, test): {col_manual: valor}}.
    """
    if not os.path.exists(QC_PATH):
        return {}
    prev = pd.read_csv(QC_PATH, sep=';', dtype=str).fillna('')
    out = {}
    for _, r in prev.iterrows():
        key = (str(r.get('subject', '')).strip().lower(),
               str(r.get('test', '')).strip())
        out[key] = {c: str(r.get(c, '')).strip() for c in MANUAL_COLS}
    return out


def main():
    previas = cargar_decisiones_previas()

    rows = []
    for s in ALL_SUBJECTS:
        for t in ANALYZED_TESTS:
            r = evaluar_ensayo(s, t)
            man = previas.get((s.lower(), t), {})
            for c in MANUAL_COLS:
                r[c] = man.get(c, '')
            rows.append(r)

    cols = (['subject', 'test', 'senal_auto', 'motivo_auto',
             'duracion_s', 'n_muestras', 'rango_max_tronco_deg'] + MANUAL_COLS)
    df = pd.DataFrame(rows)[cols]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(QC_PATH, index=False, sep=';', decimal=',')

    # -- Resumen por pantalla --------------------------------------------------
    print('=' * 78)
    print('CONTROL DE CALIDAD - ensayos de los tests 3, 7, 11 y 14')
    print('=' * 78)
    counts = df['senal_auto'].value_counts()
    print(f'Total ensayos evaluados: {len(df)}')
    for dec in ['OK', 'REVISAR', 'EXCLUIR']:
        print(f'  Senal {dec:8}: {int(counts.get(dec, 0))}')
    n_excl = int((df['decision_manual'].str.lower() == 'excluir').sum())
    n_rec  = int((df['decision_manual'].str.lower() == 'recorte').sum())
    print(f'Decision manual: {n_excl} excluir, {n_rec} recorte manual')

    for dec, titulo in [('EXCLUIR', 'Senal EXCLUIR - artefacto, sin medida valida'),
                        ('REVISAR', 'Senal REVISAR - corto, comprobar en OpenCap')]:
        sub = df[df.senal_auto == dec].sort_values(['test', 'subject'])
        if len(sub) == 0:
            continue
        print(f'\n{"-"*78}\n{titulo}\n{"-"*78}')
        print(f'{"sujeto":8}{"test":11}{"dur(s)":9}{"manual":10}{"motivo auto"}')
        for _, r in sub.iterrows():
            ds = f'{r.duracion_s:.1f}' if not pd.isna(r.duracion_s) and r.duracion_s != '' else '-'
            man = r.decision_manual if r.decision_manual else '(pend)'
            print(f'{r.subject:8}{r.test:11}{str(ds):9}{man:10}{r.motivo_auto}')

    print(f'\n{"="*78}')
    print(f'Tabla guardada -> {QC_PATH}')
    print('  Edita decision_manual / trim_inicio_s / trim_fin_s en ese archivo.')
    print('  Re-ejecutar este script refresca lo automatico y conserva lo manual.')


if __name__ == '__main__':
    main()
