"""
Test 7 — Bipedestación estática, pies juntos, ojos abiertos (test7.mot)
========================================================================
Score_inst = 0.50 × S_Mec1 + 0.40 × S_Mec2 + 0.10 × S_time

No hay fases que detectar: todo el intervalo recortado entre pisotones
es la fase de análisis.

S_Mec1  — Control del CoM / sway postural (bilateral, umbrales pies juntos)
S_Mec2  — Ankle-Hip Ratio bilateral + Asimetría tobillo + Trunk Sway ML
S_time  — min(T / 30, 1)   donde T = duración del ensayo recortado (s)
"""

import sys
import os
import numpy as np
import pandas as pd

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


# ── S_Mec1 ────────────────────────────────────────────────────────────────────

def compute_sway_features(df_trial, T):
    """
    Métricas de sway del CoM sobre todo el ensayo.
    pelvis_tz = AP (m), pelvis_tx = ML (m) → convertidas a mm.
    T en segundos.
    """
    tz_mm = df_trial['pelvis_tz'].values * 1000.0   # AP, mm
    tx_mm = df_trial['pelvis_tx'].values * 1000.0   # ML, mm

    rms_ap   = float(np.std(tz_mm))
    rms_ml   = float(np.std(tx_mm))
    range_ap = float(np.max(tz_mm) - np.min(tz_mm))
    range_ml = float(np.max(tx_mm) - np.min(tx_mm))
    sway_area = range_ap * range_ml                         # mm²

    dtz = np.diff(tz_mm)
    dtx = np.diff(tx_mm)
    path_length  = float(np.sum(np.sqrt(dtz**2 + dtx**2)))  # mm
    mean_com_vel = path_length / T if T > 0 else np.nan      # mm/s

    return dict(
        rms_ap_mm       = rms_ap,
        rms_ml_mm       = rms_ml,
        range_ap_mm     = range_ap,
        range_ml_mm     = range_ml,
        sway_area_mm2   = sway_area,
        path_length_mm  = path_length,
        mean_com_vel_mms= mean_com_vel,
    )

# Umbrales Test 7 (pies juntos): más estrictos que test 3
_T7_MEC1 = dict(
    rms_ml_mm        = (8,   20),    # (good, poor)
    rms_ap_mm        = (10,  25),
    mean_com_vel_mms = (12,  35),
    sway_area_mm2    = (200, 1200),
)

def compute_s_mec1(sw):
    return (0.35 * norm_inv(sw['rms_ml_mm'],        *_T7_MEC1['rms_ml_mm'])
          + 0.25 * norm_inv(sw['rms_ap_mm'],        *_T7_MEC1['rms_ap_mm'])
          + 0.25 * norm_inv(sw['mean_com_vel_mms'], *_T7_MEC1['mean_com_vel_mms'])
          + 0.15 * norm_inv(sw['sway_area_mm2'],    *_T7_MEC1['sway_area_mm2']))


# ── S_Mec2 ────────────────────────────────────────────────────────────────────

def compute_mec2_features(df_trial):
    """
    Ankle-Hip Ratio bilateral (media derecha+izquierda)
    Asimetría de tobillo entre piernas
    Range ML Trunk Sway (lumbar_bending)
    """
    ankle_r = df_trial['ankle_angle_r'].values
    ankle_l = df_trial['ankle_angle_l'].values
    hip_r   = df_trial['hip_flexion_r'].values
    hip_l   = df_trial['hip_flexion_l'].values

    # Señales bilaterales (media de ambos lados)
    ankle_mean = (ankle_r + ankle_l) / 2.0
    hip_mean   = (hip_r   + hip_l)   / 2.0

    theta_ankle = ankle_mean - np.mean(ankle_mean)
    theta_hip   = hip_mean   - np.mean(hip_mean)

    sd_ankle_bil = float(np.std(theta_ankle))
    sd_hip_bil   = float(np.std(theta_hip))
    ahr = sd_ankle_bil / sd_hip_bil if sd_hip_bil > 1e-6 else 1.0

    # Asimetría (diferencia de SD entre lado derecho e izquierdo)
    sd_ankle_r = float(np.std(ankle_r))
    sd_ankle_l = float(np.std(ankle_l))
    asymmetry_ankle = abs(sd_ankle_r - sd_ankle_l)

    # Range ML Trunk Sway
    lb = df_trial['lumbar_bending'].values
    range_ml_trunk = float(np.max(lb) - np.min(lb))

    return dict(
        ahr              = ahr,
        asymmetry_ankle_deg = asymmetry_ankle,
        range_ml_trunk_deg  = range_ml_trunk,
    )

def compute_s_mec2(m2):
    # AHR: mayor = mejor (norm_dir). Asimetría y Trunk: menor = mejor (norm_inv)
    return (0.50 * norm_dir(m2['ahr'],                  good=1.0, poor=0.3)
          + 0.30 * norm_inv(m2['asymmetry_ankle_deg'],  good=0.5, poor=3.0)
          + 0.20 * norm_inv(m2['range_ml_trunk_deg'],   good=3.0, poor=10.0))


# ── Per-subject analysis ──────────────────────────────────────────────────────

def analyze_subject(subject):
    """
    Carga, recorta y analiza el test7 de un sujeto.
    Devuelve un dict con features y scores, o None si el archivo no existe.
    """
    try:
        df = load_and_trim(subject, 'test7')
    except FileNotFoundError:
        print(f"  [SKIP] {subject}/test7: archivo no encontrado")
        return None
    except Exception as e:
        print(f"  [ERROR] {subject}/test7: {e}")
        return None

    T = float(df['time'].iloc[-1] - df['time'].iloc[0])
    print(f"  Duración ensayo: {T:.2f} s  ({len(df)} muestras)")

    sw = compute_sway_features(df, T)
    m2 = compute_mec2_features(df)

    S1    = compute_s_mec1(sw)
    S2    = compute_s_mec2(m2)
    Stime = float(min(T / 30.0, 1.0))
    score = 0.50 * S1 + 0.40 * S2 + 0.10 * Stime

    return dict(
        Duracion_s            = T,
        RMS_AP_mm             = sw['rms_ap_mm'],
        RMS_ML_mm             = sw['rms_ml_mm'],
        Rango_AP_mm           = sw['range_ap_mm'],
        Rango_ML_mm           = sw['range_ml_mm'],
        Area_Sway_mm2         = sw['sway_area_mm2'],
        Long_Trayectoria_mm   = sw['path_length_mm'],
        Vel_CoM_mm_s          = sw['mean_com_vel_mms'],
        Ratio_Tobillo_Cadera  = m2['ahr'],
        Asimetria_Tobillo_deg = m2['asymmetry_ankle_deg'],
        Rango_ML_Tronco_deg   = m2['range_ml_trunk_deg'],
        S_Mec1   = S1,
        S_Mec2   = S2,
        S_Tiempo = Stime,
        Score    = score,
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
                row[f'T7_{k}'] = v

        rows.append(row)

    df_new = pd.DataFrame(rows)

    # ── Actualizar CSV global ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    if os.path.exists(RESULTS_PATH):
        df_g = pd.read_csv(RESULTS_PATH, sep=';', decimal=',')
        drop = [c for c in df_g.columns if c.startswith('T7_') or c == 'Test7_Score']
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
            'T7_Duracion_s', 'T7_RMS_ML_mm', 'T7_RMS_AP_mm',
            'T7_Vel_CoM_mm_s', 'T7_Ratio_Tobillo_Cadera', 'T7_Asimetria_Tobillo_deg',
            'T7_S_Mec1', 'T7_S_Mec2', 'T7_S_Tiempo', 'T7_Score']
    show = [c for c in show if c in df_new.columns]
    print('\n' + df_new[show].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
