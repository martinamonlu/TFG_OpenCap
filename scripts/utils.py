r"""
=============================================================================
utils.py — Funciones compartidas para todos los scripts del TFG
=============================================================================

Este archivo contiene las funciones base que usan todos los scripts:
    - Encontrar rutas de archivos .mot
    - Cargar archivos .mot de OpenCap
    - Filtrar señales
    - Detectar pisotones de inicio y fin
    - Recortar el DataFrame al intervalo real del test

CÓMO USARLO EN OTROS SCRIPTS:
    import sys
    sys.path.append(r'C:\Users\marti\Desktop\TFG\scripts')
    from utils import get_mot_path, load_mot, detect_stomps, trim_mot

=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import json


# =============================================================================
# CONFIGURACIÓN GLOBAL
# =============================================================================

# Ruta raíz donde están todas las carpetas de sujetos
DATA_ROOT = r'C:\Users\marti\Desktop\TFG\data'

# Todos los sujetos del estudio
SUBJECTS_HEALTHY = [f'h{i:02d}' for i in range(1, 21)]   # h01 a h20
SUBJECTS_BLIND   = [f'b{i:02d}' for i in range(1, 21)]   # b01 a b20
ALL_SUBJECTS     = SUBJECTS_HEALTHY + SUBJECTS_BLIND

# Todos los tests del Mini-BESTest y sus archivos .mot correspondientes
ALL_TESTS = [
    'test1', 'test2', 'test3a', 'test3b',
    'test4-1', 'test4-2', 'test5-1', 'test5-2',
    'test6-1a', 'test6-2a', 'test6-1b', 'test6-2b',
    'test7', 'test8', 'test9', 'test10',
    'test11', 'test12', 'test13', 'test14-1', 'test14-2'
]

# Frecuencia de muestreo de OpenCap (Hz)
SAMPLING_RATE = 60.0

# Parámetros de detección de pisotones
STOMP_VELOCITY_THRESHOLD = 150.0   # °/s — umbral mínimo para considerar un pico como pisotón
STOMP_QUIET_WINDOW_S     = 1.0     # segundos de quietud requeridos ANTES del pisotón
STOMP_QUIET_THRESHOLD    = 0.005   # m — umbral de variación de pelvis_ty para considerar quietud


EXCEPTIONS_PATH = os.path.join(os.path.dirname(__file__), 'exceptions.json')

# =============================================================================
# RUTAS Y CARGA DE ARCHIVOS
# =============================================================================

def get_mot_path(subject: str, test: str) -> str:
    """
    Devuelve la ruta completa al archivo .mot de un sujeto y test concretos.

    La estructura de carpetas esperada es:
        DATA_ROOT / subject / OpenSimData / Kinematics / test.mot

    Args:
        subject : Código del sujeto, p.ej. 'h01' o 'b03'
        test    : Nombre del test, p.ej. 'test3a' o 'test1'

    Returns:
        Ruta completa al archivo .mot

    Raises:
        FileNotFoundError si el archivo no existe
    """
    path = os.path.join(DATA_ROOT, subject, 'OpenSimData', 'Kinematics', f'{test}.mot')

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No se encontró el archivo .mot:\n  {path}\n"
            f"  Comprueba que el sujeto '{subject}' y el test '{test}' son correctos."
        )
    return path


def load_mot(filepath: str) -> pd.DataFrame:
    """
    Carga un archivo .mot de OpenCap y devuelve un DataFrame limpio.

    El archivo .mot tiene una cabecera de texto que termina con la línea
    'endheader', seguida de los nombres de columna y los datos tabulados.

    Args:
        filepath : Ruta completa al archivo .mot

    Returns:
        DataFrame con todas las columnas cinemáticas y la columna 'time'
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Encontrar la línea donde terminan las cabeceras
    header_end = next(
        (i for i, line in enumerate(lines) if line.strip() == 'endheader'),
        None
    )
    if header_end is None:
        raise ValueError(f"No se encontró 'endheader' en el archivo: {filepath}")

    # Leer desde la línea de nombres de columna (header_end + 1)
    df = pd.read_csv(filepath, sep='\t', skiprows=header_end + 1)
    df.columns = [col.strip() for col in df.columns]

    return df


# =============================================================================
# FILTRADO DE SEÑALES
# =============================================================================

def lowpass_filter(signal: np.ndarray, cutoff: float = 6.0,
                   fs: float = SAMPLING_RATE, order: int = 4) -> np.ndarray:
    """
    Aplica un filtro Butterworth paso bajo a una señal.

    Args:
        signal : Array 1D con la señal a filtrar
        cutoff : Frecuencia de corte en Hz (por defecto 6 Hz)
        fs     : Frecuencia de muestreo en Hz
        order  : Orden del filtro

    Returns:
        Señal filtrada
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)


# =============================================================================
# DETECCIÓN DE PISOTONES
# =============================================================================
def load_exceptions():
    if os.path.exists(EXCEPTIONS_PATH):
        with open(EXCEPTIONS_PATH, 'r') as f:
            return json.load(f)
    return {}


def detect_stomps(df: pd.DataFrame, fs: float = SAMPLING_RATE,
                  plot: bool = False, subject: str = '', test: str = '') -> tuple:
    """
    Detecta los dos pisotones de inicio y fin del test en la señal ankle_angle_l.

    Estrategia:
        1. Calcular la velocidad angular absoluta de ankle_angle_l (derivada)
        2. Buscar picos por encima de STOMP_VELOCITY_THRESHOLD (°/s)
        3. De esos picos, quedarse solo con los que están precedidos de al
           menos STOMP_QUIET_WINDOW_S segundos de quietud en pelvis_ty
        4. El primer pico válido = pisotón de inicio
           El último pico válido = pisotón de fin

    Args:
        df      : DataFrame completo del archivo .mot
        fs      : Frecuencia de muestreo en Hz
        plot    : Si True, muestra una gráfica de verificación
        subject : Código del sujeto (solo para el título del plot)
        test    : Nombre del test (solo para el título del plot)

    Returns:
        Tupla (idx_start, idx_end) con los índices de muestra de inicio y fin
        Devuelve (0, len(df)-1) si no se detectan los pisotones correctamente
    """
    # Consultar excepciones — algunos sujetos dieron el pisotón con el pie incorrecto
    exceptions = load_exceptions()
    stomp_foot = exceptions.get(subject, {}).get(test, {}).get('stomp_foot', 'left')
    stomp_col  = 'ankle_angle_r' if stomp_foot == 'right' else 'ankle_angle_l'

    ankle    = df[stomp_col].values
    pelvis_y = df['pelvis_ty'].values
    time     = df['time'].values
    dt       = 1.0 / fs

    # --- Paso 1: velocidad angular absoluta del tobillo izquierdo ---
    d_ankle = np.abs(np.gradient(ankle, dt))

    # Suavizar ligeramente para eliminar ruido puntual pero conservar picos reales
    d_ankle_smooth = lowpass_filter(d_ankle, cutoff=10.0, fs=fs)

    # --- Paso 2: encontrar todos los picos por encima del umbral ---
    # min_distance: los pisotones están separados al menos 2 segundos
    peaks, properties = find_peaks(
        d_ankle_smooth,
        height=STOMP_VELOCITY_THRESHOLD,
        distance=int(2.0 * fs)
    )

    if len(peaks) < 2:
        print(f"  [AVISO] {subject}/{test}: Solo se detectaron {len(peaks)} pico(s). "
              f"Bajando el umbral de detección...")
        # Intentar con umbral más bajo
        peaks, _ = find_peaks(
            d_ankle_smooth,
            height=STOMP_VELOCITY_THRESHOLD * 0.6,
            distance=int(2.0 * fs)
        )

    # --- Paso 3: filtrar por quietud previa en pelvis_ty ---
    quiet_samples = int(STOMP_QUIET_WINDOW_S * fs)
    valid_peaks = []

    for peak_idx in peaks:
        # Comprobar que hay suficientes muestras previas para analizar quietud
        if peak_idx < quiet_samples:
            continue

        # Variación de pelvis_ty en la ventana previa al pico
        window_before = pelvis_y[peak_idx - quiet_samples : peak_idx]
        variation = np.max(window_before) - np.min(window_before)

        if variation < STOMP_QUIET_THRESHOLD:
            valid_peaks.append(peak_idx)

    # --- Paso 4: seleccionar inicio y fin ---
    if len(valid_peaks) >= 2:
        idx_start = valid_peaks[0]
        idx_end   = valid_peaks[-1]
        status = 'OK'
    elif len(valid_peaks) == 1:
        # Solo un pisotón detectado con quietud — usar el primero o último pico general
        print(f"  [AVISO] {subject}/{test}: Solo 1 pisotón con quietud previa detectado. "
              f"Usando picos generales como fallback.")
        idx_start = valid_peaks[0]
        idx_end   = peaks[-1] if peaks[-1] != valid_peaks[0] else len(df) - 1
        status = 'PARCIAL'
    else:
        print(f"  [AVISO] {subject}/{test}: No se detectaron pisotones con quietud previa. "
              f"Usando inicio y fin del archivo.")
        idx_start = 0
        idx_end   = len(df) - 1
        status = 'FALLIDO'

    # Pequeño margen tras el pisotón para no incluir el impacto en el análisis
    # (0.3 s después del pisotón de inicio, 0.3 s antes del pisotón de fin)
    margin = int(0.3 * fs)
    idx_start_trim = min(idx_start + margin, len(df) - 1)
    idx_end_trim   = max(idx_end   - margin, 0)

    print(f"  [{status}] {subject}/{test}: "
          f"inicio = {time[idx_start]:.2f} s -> {time[idx_start_trim]:.2f} s | "
          f"fin = {time[idx_end]:.2f} s -> {time[idx_end_trim]:.2f} s | "
          f"duración recortada = {time[idx_end_trim] - time[idx_start_trim]:.2f} s")

    # --- Plot de verificación opcional ---
    if plot:
        fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f'Detección de pisotones — {subject} / {test}', fontsize=12)

        axes[0].plot(time, ankle, color='steelblue', linewidth=0.8)
        axes[0].axvline(time[idx_start], color='green', linewidth=2,
                        linestyle='--', label='Pisotón inicio')
        axes[0].axvline(time[idx_end],   color='red',   linewidth=2,
                        linestyle='--', label='Pisotón fin')
        axes[0].axvspan(time[idx_start_trim], time[idx_end_trim],
                        alpha=0.10, color='green', label='Intervalo análisis')
        axes[0].set_ylabel(f'{stomp_col} (°)')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(time, d_ankle_smooth, color='tomato', linewidth=0.8)
        axes[1].axhline(STOMP_VELOCITY_THRESHOLD, color='orange',
                        linewidth=1, linestyle=':', label=f'Umbral {STOMP_VELOCITY_THRESHOLD} °/s')
        axes[1].scatter(time[list(peaks)], d_ankle_smooth[list(peaks)],
                        color='gray', s=30, zorder=4, label='Todos los picos')
        if valid_peaks:
            axes[1].scatter(time[valid_peaks], d_ankle_smooth[valid_peaks],
                            color='green', s=60, zorder=5, label='Picos válidos (con quietud)')
        axes[1].set_ylabel('|d/dt ankle_angle_l| (°/s)')
        axes[1].legend(fontsize=8)
        axes[1].grid(True, alpha=0.3)

        axes[2].plot(time, pelvis_y, color='seagreen', linewidth=0.8)
        axes[2].axvline(time[idx_start], color='green', linewidth=2, linestyle='--')
        axes[2].axvline(time[idx_end],   color='red',   linewidth=2, linestyle='--')
        axes[2].set_ylabel('pelvis_ty (m)')
        axes[2].set_xlabel('Tiempo (s)')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return idx_start_trim, idx_end_trim


# =============================================================================
# RECORTE DEL DATAFRAME
# =============================================================================

def trim_mot(df: pd.DataFrame, idx_start: int, idx_end: int,
             reset_time: bool = True) -> pd.DataFrame:
    """
    Recorta el DataFrame al intervalo real del test (entre los dos pisotones).

    Args:
        df          : DataFrame completo del archivo .mot
        idx_start   : Índice de muestra de inicio (después del pisotón inicial)
        idx_end     : Índice de muestra de fin (antes del pisotón final)
        reset_time  : Si True, resetea el tiempo para que empiece en 0

    Returns:
        DataFrame recortado con el tiempo reseteado a 0
    """
    df_trimmed = df.iloc[idx_start:idx_end].copy()
    df_trimmed = df_trimmed.reset_index(drop=True)

    if reset_time:
        df_trimmed['time'] = df_trimmed['time'] - df_trimmed['time'].iloc[0]

    return df_trimmed


# =============================================================================
# FUNCIÓN DE CONVENIENCIA: cargar + detectar + recortar en un solo paso
# =============================================================================

def load_and_trim(subject: str, test: str,
                  plot: bool = False) -> pd.DataFrame:
    """
    Carga el archivo .mot de un sujeto y test, detecta los pisotones
    y devuelve el DataFrame ya recortado al intervalo real del ejercicio.

    Es la función principal que usarán todos los scripts de análisis.

    Ejemplo de uso:
        df = load_and_trim('h01', 'test3a')

    Args:
        subject : Código del sujeto, p.ej. 'h01'
        test    : Nombre del test, p.ej. 'test3a'
        plot    : Si True, muestra la gráfica de detección de pisotones

    Returns:
        DataFrame recortado y con tiempo reseteado a 0
    """
    filepath = get_mot_path(subject, test)
    df       = load_mot(filepath)
    idx_start, idx_end = detect_stomps(df, plot=plot, subject=subject, test=test)
    df_trimmed = trim_mot(df, idx_start, idx_end)
    return df_trimmed
