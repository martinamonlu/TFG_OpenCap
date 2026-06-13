r"""
=============================================================================
trim_all.py — Recorta todos los archivos .mot de todos los sujetos
=============================================================================

DESCRIPCIÓN:
    Este es el PRIMER script que hay que ejecutar, antes de cualquier análisis.

    Para cada sujeto y cada test:
        1. Carga el archivo .mot original de OpenCap
        2. Detecta los dos pisotones (inicio y fin del ejercicio)
        3. Recorta el DataFrame al intervalo real del test
        4. Guarda el archivo recortado en una carpeta 'trimmed' dentro
           de la carpeta del sujeto

    Los archivos recortados se guardan con el mismo nombre pero en:
        TFG/data/h01/OpenSimData/Kinematics/trimmed/test3-a.mot

    Todos los scripts de análisis posteriores leerán de la carpeta 'trimmed'.

CÓMO EJECUTARLO:
    cd C:\Users\marti\Desktop\TFG\scripts
    python3 trim_all.py

SALIDA:
    - Archivos .mot recortados en cada carpeta trimmed/
    - Un CSV resumen en TFG/results/trim_log.csv con el resultado
      de cada detección (OK, PARCIAL o FALLIDO) para poder
      revisar manualmente los casos problemáticos

=============================================================================
"""

import os
import sys
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Añadir la carpeta scripts al path para poder importar utils.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils import (ALL_SUBJECTS, ALL_TESTS, DATA_ROOT,
                   get_mot_path, load_mot, detect_stomps, detect_stomps_tug,
                   trim_mot, get_manual_trim)


# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Si True, muestra la gráfica de detección para CADA archivo (muy lento,
# solo usar para depurar casos concretos)
PLOT_ALL = False

# Si True, muestra la gráfica solo cuando la detección es PARCIAL o FALLIDA
PLOT_ON_FAILURE = True

# Carpeta donde se guardan los archivos recortados (dentro de cada sujeto)
TRIMMED_SUBDIR = os.path.join('OpenSimData', 'Kinematics', 'trimmed')

# Carpeta de resultados globales
RESULTS_DIR = os.path.join(DATA_ROOT, '..', 'results')


# =============================================================================
# FUNCIÓN PARA GUARDAR UN .mot RECORTADO
# =============================================================================

def save_trimmed_mot(df_trimmed: pd.DataFrame, subject: str, test: str):
    """
    Guarda el DataFrame recortado como archivo .mot en la carpeta trimmed/.

    Conserva el formato original de OpenCap (cabecera + datos tabulados).

    Args:
        df_trimmed : DataFrame ya recortado
        subject    : Código del sujeto
        test       : Nombre del test
    """
    # Crear la carpeta trimmed si no existe
    trimmed_dir = os.path.join(DATA_ROOT, subject, TRIMMED_SUBDIR)
    os.makedirs(trimmed_dir, exist_ok=True)

    output_path = os.path.join(trimmed_dir, f'{test}.mot')

    # Escribir cabecera compatible con OpenCap/OpenSim
    n_rows = len(df_trimmed)
    n_cols = len(df_trimmed.columns)

    with open(output_path, 'w') as f:
        f.write('Coordinates\n')
        f.write('version=1\n')
        f.write(f'nRows={n_rows}\n')
        f.write(f'nColumns={n_cols}\n')
        f.write('inDegrees=yes\n')
        f.write('\n')
        f.write('Units are S.I. units (second, meters, Newtons, ...)\n')
        f.write('If the header above contains a line with \'inDegrees\', '
                'this indicates whether rotational values are in degrees (yes) or radians (no).\n')
        f.write('\n')
        f.write('endheader\n')

        # Escribir nombres de columnas y datos
        df_trimmed.to_csv(f, sep='\t', index=False, float_format='%.8f')

    return output_path


# =============================================================================
# PROCESAMIENTO PRINCIPAL
# =============================================================================

def process_subject(subject: str, log_rows: list):
    """
    Procesa todos los tests de un sujeto: carga, detecta pisotones,
    recorta y guarda.

    Args:
        subject  : Código del sujeto, p.ej. 'h01'
        log_rows : Lista donde se van añadiendo las filas del log CSV
    """
    print(f"\n{'='*60}")
    print(f"  Sujeto: {subject}")
    print(f"{'='*60}")

    for test in ALL_TESTS:
        # Intentar cargar el archivo — puede que algún sujeto no tenga todos los tests
        try:
            filepath = get_mot_path(subject, test)
        except FileNotFoundError:
            print(f"  [SKIP] {test}: archivo no encontrado, saltando.")
            log_rows.append({
                'subject': subject,
                'test'   : test,
                'status' : 'NO_FILE',
                'duration_s': None,
                'start_s'   : None,
                'end_s'     : None,
            })
            continue

        try:
            # Cargar
            df = load_mot(filepath)

            # Recorte manual (revision_manual.csv) tiene prioridad: para
            # ensayos cuyos pisotones el detector automatico no capta.
            manual = get_manual_trim(subject, test)
            is_tug = test.startswith('test14')
            if manual is not None:
                ini_s, fin_s = manual
                time_arr  = df['time'].values
                idx_start = int(np.searchsorted(time_arr, ini_s))
                idx_end   = int(min(np.searchsorted(time_arr, fin_s), len(df) - 1))
                print(f"  [MANUAL] {subject}/{test}: recorte fijado a mano "
                      f"{ini_s:.2f}s -> {fin_s:.2f}s")
            elif is_tug:
                # El TUG (test14) usa un detector especifico (sin filtro de
                # quietud, ver utils.detect_stomps_tug)
                idx_start, idx_end = detect_stomps_tug(
                    df, subject=subject, test=test)
            else:
                idx_start, idx_end = detect_stomps(
                    df, plot=PLOT_ALL, subject=subject, test=test)

            # Determinar el estado de la detección para el log
            time = df['time'].values
            duration = time[idx_end] - time[idx_start]

            if manual is not None:
                status = 'MANUAL'
            elif idx_start == 0 and idx_end == len(df) - 1:
                status = 'FALLIDO'
            elif idx_start == 0 or idx_end == len(df) - 1:
                status = 'PARCIAL'
            else:
                status = 'OK'

            # Si la detección falló y el usuario quiere ver el plot, mostrarlo
            # (solo para el detector estándar; detect_stomps_tug no tiene plot)
            if PLOT_ON_FAILURE and status not in ('OK', 'MANUAL') and not is_tug:
                print(f"  [!] Mostrando gráfica de verificación para {subject}/{test}...")
                detect_stomps(df, plot=True, subject=subject, test=test)

            # Recortar
            df_trimmed = trim_mot(df, idx_start, idx_end, reset_time=True)

            # Guardar
            output_path = save_trimmed_mot(df_trimmed, subject, test)

            print(f"  [GUARDADO] → {os.path.relpath(output_path, DATA_ROOT)}")

            log_rows.append({
                'subject'   : subject,
                'test'      : test,
                'status'    : status,
                'duration_s': round(duration, 2),
                'start_s'   : round(time[idx_start], 2),
                'end_s'     : round(time[idx_end], 2),
            })

        except Exception as e:
            print(f"  [ERROR] {subject}/{test}: {e}")
            log_rows.append({
                'subject'   : subject,
                'test'      : test,
                'status'    : 'ERROR',
                'duration_s': None,
                'start_s'   : None,
                'end_s'     : None,
            })


def main():
    print('\n' + '='*60)
    print('  00_trim_all.py — Recorte de archivos .mot')
    print(f'  Sujetos: {len(ALL_SUBJECTS)} ({ALL_SUBJECTS[0]} → {ALL_SUBJECTS[-1]})')
    print(f'  Tests por sujeto: {len(ALL_TESTS)}')
    print(f'  Total archivos a procesar: {len(ALL_SUBJECTS) * len(ALL_TESTS)}')
    print('='*60)

    # Crear carpeta de resultados si no existe
    os.makedirs(RESULTS_DIR, exist_ok=True)

    log_rows = []

    for subject in ALL_SUBJECTS:
        # Comprobar que la carpeta del sujeto existe
        subject_dir = os.path.join(DATA_ROOT, subject)
        if not os.path.exists(subject_dir):
            print(f"\n[SKIP] Carpeta no encontrada para sujeto '{subject}', saltando.")
            continue

        process_subject(subject, log_rows)

    # Guardar log CSV
    log_path = os.path.join(RESULTS_DIR, 'trim_log.csv')
    df_log = pd.DataFrame(log_rows)
    df_log.to_csv(log_path, index=False)

    # Resumen final
    print(f"\n{'='*60}")
    print('  RESUMEN FINAL')
    print(f"{'='*60}")
    status_counts = df_log['status'].value_counts()
    for status, count in status_counts.items():
        print(f"  {status:10s}: {count} archivos")

    # Mostrar los casos que necesitan revisión manual
    problemas = df_log[df_log['status'].isin(['PARCIAL', 'FALLIDO', 'ERROR'])]
    if len(problemas) > 0:
        print(f"\n  ⚠ CASOS QUE REQUIEREN REVISIÓN MANUAL ({len(problemas)}):")
        for _, row in problemas.iterrows():
            print(f"    - {row['subject']} / {row['test']} → {row['status']}")
        print(f"\n  Revisa estos casos en el log: {log_path}")
    else:
        print('\n  ✓ Todos los archivos detectados correctamente.')

    print(f"\n  Log completo guardado en: {log_path}")
    print('='*60 + '\n')


if __name__ == '__main__':
    main()
