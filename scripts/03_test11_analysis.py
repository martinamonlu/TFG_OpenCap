"""
Test 11 — Marcha con giros de cabeza horizontal (test11.mot)
=============================================================
Score_inst = 0.45 × S_Mec1 + 0.35 × S_Mec2 + 0.20 × S_Mec3

No hay S_time: lo relevante es cómo camina, no cuánto aguanta.
Todo el intervalo recortado entre pisotones es marcha activa.

S_Mec1  — Velocidad y consistencia de la marcha
S_Mec2  — Estabilidad de la trayectoria (lateral + oscilación tronco)
S_Mec3  — Estabilidad global del tronco (rangos ML, AP, rotación)

Detección de pasos: picos en pelvis_ty (mín. 0.3 s entre pasos).
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

sys.path.append(r'C:\Users\marti\Desktop\TFG\scripts')
from utils import ALL_SUBJECTS, SAMPLING_RATE, load_and_trim

RESULTS_PATH = r'C:\Users\marti\Desktop\TFG\results\resultados_globales.csv'
FS = SAMPLING_RATE


# ── Normalisation helpers ─────────────────────────────────────────────────────

def norm_inv(val, good, poor):
    """Menor = mejor → 1 cuando val≤good, 0 cuando val≥poor."""
    if poor == good:
        return 0.0
    return float(np.clip((poor - val) / (poor - good), 0.0, 1.0))

def norm_dir(val, good, poor):
    """Mayor = mejor → 1 cuando val≥good, 0 cuando val≤poor."""
    if good == poor:
        return 0.0
    return float(np.clip((val - poor) / (good - poor), 0.0, 1.0))


# ── Step detection ────────────────────────────────────────────────────────────

def detect_steps(df, fs=FS):
    """
    Detecta pasos como picos en pelvis_ty (mínimo 0.3 s entre pasos).
    Devuelve (step_indices, step_times, intervalos_paso).
    """
    pelvis_ty = df['pelvis_ty'].values
    time      = df['time'].values

    min_distance = int(0.3 * fs)
    step_idx, _ = find_peaks(pelvis_ty, distance=min_distance)

    if len(step_idx) < 2:
        return step_idx, time[step_idx] if len(step_idx) else np.array([]), np.array([])

    step_times      = time[step_idx]
    intervalos_paso = np.diff(step_times)   # tiempo entre pasos consecutivos (s)

    return step_idx, step_times, intervalos_paso


# ── S_Mec1: velocidad y consistencia ─────────────────────────────────────────

def compute_mec1_features(df, step_idx, intervalos_paso):
    """
    Velocidad media, variabilidad de velocidad y variabilidad de cadencia.
    """
    pelvis_tz = df['pelvis_tz'].values           # dirección de marcha (m)
    time      = df['time'].values
    T_total   = float(time[-1] - time[0])
    dt        = 1.0 / FS

    distancia_total = float(np.max(pelvis_tz) - np.min(pelvis_tz))   # m
    vel_media       = distancia_total / T_total if T_total > 0 else np.nan

    # Velocidad instantánea (diferencias finitas)
    vel_instant = np.abs(np.diff(pelvis_tz)) / dt                    # m/s
    sd_vel      = float(np.std(vel_instant))

    # Variabilidad de cadencia (CV de intervalos entre pasos)
    if len(intervalos_paso) >= 2:
        mean_int = float(np.mean(intervalos_paso))
        cv_cad   = float(np.std(intervalos_paso) / mean_int * 100) if mean_int > 0 else np.nan
        cadencia = float(len(step_idx) / T_total * 60) if T_total > 0 else np.nan
    else:
        cv_cad   = np.nan
        cadencia = np.nan

    return dict(
        vel_media_ms  = vel_media,
        sd_vel_ms     = sd_vel,
        cadencia_spm  = cadencia,
        cv_cadencia_pct = cv_cad,
        distancia_total_m = distancia_total,
        T_total_s     = T_total,
        n_pasos       = len(step_idx),
    )

def compute_s_mec1(m1):
    cv = m1['cv_cadencia_pct'] if not np.isnan(m1.get('cv_cadencia_pct', np.nan)) else 10.0
    return (0.40 * norm_dir(m1['vel_media_ms'], good=1.0,  poor=0.5)
          + 0.35 * norm_inv(m1['sd_vel_ms'],    good=0.05, poor=0.20)
          + 0.25 * norm_inv(cv,                 good=3.0,  poor=10.0))


# ── S_Mec2: estabilidad de trayectoria ───────────────────────────────────────

def compute_mec2_features(df):
    """
    Desviación lateral RMS (residuos de regresión lineal AP→ML).
    Trunk ML RMS y Trunk AP RMS.
    """
    pelvis_tz = df['pelvis_tz'].values          # AP (dirección de marcha)
    pelvis_tx = df['pelvis_tx'].values          # ML (lateral)

    # Regresión lineal AP→ML para extraer desviación lateral errática
    coeffs    = np.polyfit(pelvis_tz, pelvis_tx, 1)
    linea_rec = np.polyval(coeffs, pelvis_tz)
    residuos  = pelvis_tx - linea_rec
    dev_lat_rms_mm = float(np.sqrt(np.mean(residuos**2)) * 1000.0)   # m → mm

    # Oscilación lateral y AP del tronco
    lb  = df['lumbar_bending'].values
    lex = df['lumbar_extension'].values
    trunk_ml_rms = float(np.std(lb))    # °
    trunk_ap_rms = float(np.std(lex))   # °

    return dict(
        dev_lat_rms_mm = dev_lat_rms_mm,
        trunk_ml_rms_deg = trunk_ml_rms,
        trunk_ap_rms_deg = trunk_ap_rms,
    )

def compute_s_mec2(m2):
    return (0.50 * norm_inv(m2['dev_lat_rms_mm'],   good=20.0, poor=80.0)
          + 0.30 * norm_inv(m2['trunk_ml_rms_deg'], good=3.0,  poor=10.0)
          + 0.20 * norm_inv(m2['trunk_ap_rms_deg'], good=3.0,  poor=10.0))


# ── S_Mec3: estabilidad global del tronco ────────────────────────────────────

def compute_mec3_features(df):
    """
    Rangos de movimiento del tronco en ML, AP y rotación.
    """
    lb  = df['lumbar_bending'].values
    lex = df['lumbar_extension'].values
    lro = df['lumbar_rotation'].values

    range_ml_trunk  = float(np.max(lb)  - np.min(lb))
    range_ap_trunk  = float(np.max(lex) - np.min(lex))
    range_rot_trunk = float(np.max(lro) - np.min(lro))

    return dict(
        range_ml_trunk_deg  = range_ml_trunk,
        range_ap_trunk_deg  = range_ap_trunk,
        range_rot_trunk_deg = range_rot_trunk,
    )

def compute_s_mec3(m3):
    # Umbrales empíricos: good = p25 healthy, poor = p75 blind (muestra propia, n=40)
    # ML:  good=17.2°, poor=21.0° | AP: good=15.1°, poor=28.0° | Rot: good=15.2°, poor=20.8°
    return (0.40 * norm_inv(m3['range_ml_trunk_deg'],  good=17.2, poor=21.0)
          + 0.35 * norm_inv(m3['range_ap_trunk_deg'],  good=15.1, poor=28.0)
          + 0.25 * norm_inv(m3['range_rot_trunk_deg'], good=15.2, poor=20.8))


# ── Per-subject analysis ──────────────────────────────────────────────────────

def analyze_subject(subject):
    """
    Carga, recorta y analiza el test11 de un sujeto.
    Devuelve un dict con features y scores, o None si el archivo no existe.
    """
    try:
        df = load_and_trim(subject, 'test11')
    except FileNotFoundError:
        print(f"  [SKIP] {subject}/test11: archivo no encontrado")
        return None
    except Exception as e:
        print(f"  [ERROR] {subject}/test11: {e}")
        return None

    T_total = float(df['time'].iloc[-1] - df['time'].iloc[0])
    print(f"  Duración ensayo: {T_total:.2f} s  ({len(df)} muestras)")

    # Detección de pasos
    step_idx, step_times, intervalos_paso = detect_steps(df)
    print(f"  Pasos detectados: {len(step_idx)}")

    if len(step_idx) < 4:
        print(f"  [AVISO] Muy pocos pasos detectados ({len(step_idx)}) "
              f"— cadencia y CV pueden ser poco fiables")

    m1 = compute_mec1_features(df, step_idx, intervalos_paso)
    m2 = compute_mec2_features(df)
    m3 = compute_mec3_features(df)

    S1 = compute_s_mec1(m1)
    S2 = compute_s_mec2(m2)
    S3 = compute_s_mec3(m3)
    score = 0.45 * S1 + 0.35 * S2 + 0.20 * S3

    return dict(
        Duracion_s             = m1['T_total_s'],
        N_Pasos                = m1['n_pasos'],
        Distancia_Total_m      = m1['distancia_total_m'],
        Vel_Marcha_m_s         = m1['vel_media_ms'],
        SD_Vel_Marcha_m_s      = m1['sd_vel_ms'],
        Cadencia_pasos_min     = m1['cadencia_spm'],
        CV_Cadencia_pct        = m1['cv_cadencia_pct'],
        Desv_Lateral_RMS_mm    = m2['dev_lat_rms_mm'],
        RMS_ML_Tronco_deg      = m2['trunk_ml_rms_deg'],
        RMS_AP_Tronco_deg      = m2['trunk_ap_rms_deg'],
        Rango_ML_Tronco_deg    = m3['range_ml_trunk_deg'],
        Rango_AP_Tronco_deg    = m3['range_ap_trunk_deg'],
        Rango_Rot_Tronco_deg   = m3['range_rot_trunk_deg'],
        S_Mec1 = S1,
        S_Mec2 = S2,
        S_Mec3 = S3,
        Score  = score,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows = []

    for subject in ALL_SUBJECTS:
        print(f"\n{'='*45}\n{subject}")
        group = 'healthy' if subject.startswith('h') else 'blind'
        row   = {'subject': subject, 'group': group}

        res = analyze_subject(subject)
        if res is not None:
            for k, v in res.items():
                row[f'T11_{k}'] = v

        rows.append(row)

    df_new = pd.DataFrame(rows)

    # ── Actualizar CSV global ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    if os.path.exists(RESULTS_PATH):
        df_g = pd.read_csv(RESULTS_PATH, sep=';', decimal=',')
        drop = [c for c in df_g.columns if c.startswith('T11_') or c == 'Test11_Score']
        df_g = df_g.drop(columns=drop, errors='ignore')
        merge_cols = [c for c in df_new.columns if c not in ('group',)]
        df_g = df_g.merge(df_new[merge_cols], on='subject', how='outer')
        df_g['group'] = df_g['group'].fillna(
            df_g['subject'].apply(lambda s: 'healthy' if s.startswith('h') else 'blind'))
    else:
        df_g = df_new

    df_g.to_csv(RESULTS_PATH, index=False, sep=';', decimal=',')
    print(f"\n✓ Resultados guardados → {RESULTS_PATH}")

    # ── Resumen por pantalla ──────────────────────────────────────────────────
    show = ['subject', 'group',
            'T11_Vel_Marcha_m_s', 'T11_SD_Vel_Marcha_m_s', 'T11_CV_Cadencia_pct',
            'T11_Desv_Lateral_RMS_mm',
            'T11_Rango_ML_Tronco_deg', 'T11_Rango_AP_Tronco_deg',
            'T11_S_Mec1', 'T11_S_Mec2', 'T11_S_Mec3', 'T11_Score']
    show = [c for c in show if c in df_new.columns]
    print('\n' + df_new[show].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
