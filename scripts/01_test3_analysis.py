"""
Test 3 — Apoyo monopodal (test3a.mot + test3b.mot)
======================================================
Score_inst = 0.40 × S_Mec1 + 0.30 × S_Mec2 + 0.30 × S_time

Fase 2 (análisis): knee_angle_l > 10° sostenido ≥ 0.3 s
         hasta que knee_angle_l vuelve a bajar de 10°.
         Se toma el período más largo detectado.

S_Mec1  — Control del CoM en base reducida
S_Mec2  — Ankle-Hip Ratio + signo de Trendelenburg + Trunk Sway ML
S_time  — min(T / 20, 1)   donde T = duración real de Fase 2 (s)
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


# ── Phase 2 detection ─────────────────────────────────────────────────────────

def detect_phase2(df, fs=FS, knee_col='knee_angle_l'):
    """
    Devuelve (idx_start, idx_end) del período más largo donde
    knee_col > 10° sostenido >= 0.3 s.
    Devuelve (None, None) si no se encuentra ninguno.
    """
    THRESHOLD  = 10.0           # grados
    MIN_SUST_S = 0.3            # segundos
    min_samp   = int(MIN_SUST_S * fs)

    knee  = df[knee_col].values
    above = knee > THRESHOLD

    best_start, best_end, best_len = None, None, 0
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            run_len = j - i
            if run_len >= min_samp and run_len > best_len:
                best_start, best_end, best_len = i, j - 1, run_len
            i = j
        else:
            i += 1

    return best_start, best_end


# ── S_Mec1 ────────────────────────────────────────────────────────────────────

def compute_sway_features(df_phase, T):
    """
    Métricas de sway del CoM sobre la Fase 2.
    pelvis_tz = AP (m), pelvis_tx = ML (m) → convertidas a mm.
    T en segundos.
    """
    # Detrend para eliminar drift lento (p.ej. sujeto moviéndose levemente)
    # np.polyfit grado 1 elimina tendencia lineal, dejando solo la oscilación real
    tz_raw = df_phase['pelvis_tz'].values * 1000.0
    tx_raw = df_phase['pelvis_tx'].values * 1000.0
    t_idx  = np.arange(len(tz_raw), dtype=float)
    tz_mm  = tz_raw - np.polyval(np.polyfit(t_idx, tz_raw, 1), t_idx)
    tx_mm  = tx_raw - np.polyval(np.polyfit(t_idx, tx_raw, 1), t_idx)

    rms_ap   = float(np.std(tz_mm))
    rms_ml   = float(np.std(tx_mm))
    range_ap = float(np.max(tz_mm) - np.min(tz_mm))
    range_ml = float(np.max(tx_mm) - np.min(tx_mm))
    sway_area = range_ap * range_ml                 # mm²

    dtz = np.diff(tz_mm)
    dtx = np.diff(tx_mm)
    path_length   = float(np.sum(np.sqrt(dtz**2 + dtx**2)))   # mm
    mean_com_vel  = path_length / T if T > 0 else np.nan       # mm/s

    return dict(
        rms_ap_mm      = rms_ap,
        rms_ml_mm      = rms_ml,
        range_ap_mm    = range_ap,
        range_ml_mm    = range_ml,
        sway_area_mm2  = sway_area,
        path_length_mm = path_length,
        mean_com_vel_mms = mean_com_vel,
    )

# Umbrales Test 3 (monopodal): good → 1, poor → 0
_T3_MEC1 = dict(
    rms_ml_mm       = (10,  25),    # (good, poor)
    rms_ap_mm       = (12,  30),
    mean_com_vel_mms= (15,  40),
    sway_area_mm2   = (300, 1500),
)

def compute_s_mec1(sw):
    return (0.35 * norm_inv(sw['rms_ml_mm'],       *_T3_MEC1['rms_ml_mm'])
          + 0.25 * norm_inv(sw['rms_ap_mm'],       *_T3_MEC1['rms_ap_mm'])
          + 0.25 * norm_inv(sw['mean_com_vel_mms'],*_T3_MEC1['mean_com_vel_mms'])
          + 0.15 * norm_inv(sw['sway_area_mm2'],   *_T3_MEC1['sway_area_mm2']))


# ── S_Mec2 ────────────────────────────────────────────────────────────────────

def compute_mec2_features(df_phase, support_side='r'):
    """
    Ankle-Hip Ratio (pierna de apoyo: 'r' para test3a, 'l' para test3b)
    Mean Pelvic Drop (signo de Trendelenburg, pelvis_list)
    Range ML Trunk Sway (lumbar_bending)
    """
    # Ankle-Hip Ratio — usar la pierna de apoyo (opuesta a la que se levanta)
    sd_ankle = float(np.std(df_phase[f'ankle_angle_{support_side}'].values))
    sd_hip   = float(np.std(df_phase[f'hip_flexion_{support_side}'].values))
    ahr = sd_ankle / sd_hip if sd_hip > 1e-6 else 1.0

    # Mean Pelvic Drop: referencia = media primeros 0.5 s de Fase 2
    if 'pelvis_list' in df_phase.columns:
        pl     = df_phase['pelvis_list'].values
        n_ref  = max(1, int(0.5 * FS))
        ref    = float(np.mean(pl[:n_ref]))
        pelvic_drop = float(abs(np.mean(pl - ref)))
    else:
        pelvic_drop = 0.0

    # Range ML Trunk Sway
    lb = df_phase['lumbar_bending'].values
    range_ml_trunk = float(np.max(lb) - np.min(lb))

    return dict(
        ahr              = ahr,
        pelvic_drop_deg  = pelvic_drop,
        range_ml_trunk_deg = range_ml_trunk,
    )

def compute_s_mec2(m2):
    # AHR: mayor = mejor (norm_dir).  Drop y Trunk: menor = mejor (norm_inv)
    return (0.50 * norm_dir(m2['ahr'],               good=1.0, poor=0.3)
          + 0.30 * norm_inv(m2['pelvic_drop_deg'],   good=2.0, poor=8.0)
          + 0.20 * norm_inv(m2['range_ml_trunk_deg'],good=5.0, poor=15.0))


# ── Per-trial analysis ────────────────────────────────────────────────────────

def analyze_trial(subject, trial):
    """
    Carga, recorta y analiza un ensayo de Test 3.
    Devuelve un dict con todas las features y scores, o None si el archivo no existe.
    """
    try:
        df = load_and_trim(subject, trial)
    except FileNotFoundError:
        print(f"  [SKIP] {subject}/{trial}: archivo no encontrado")
        return None
    except Exception as e:
        print(f"  [ERROR] {subject}/{trial}: {e}")
        return None

    # test3a: levanta pierna izq → apoyo derecho; test3b: levanta pierna der → apoyo izquierdo
    knee_col     = 'knee_angle_r' if trial == 'test3b' else 'knee_angle_l'
    support_side = 'l'            if trial == 'test3b' else 'r'
    idx_s, idx_e = detect_phase2(df, knee_col=knee_col)
    if idx_s is None:
        print(f"  [AVISO] {subject}/{trial}: Fase 2 no detectada -> usando ensayo completo")
        idx_s, idx_e = 0, len(df) - 1

    df2 = df.iloc[idx_s : idx_e + 1].reset_index(drop=True)
    T   = float(df['time'].iloc[idx_e] - df['time'].iloc[idx_s])
    print(f"  Fase 2: {T:.2f} s  ({idx_s}->{idx_e}, {len(df2)} muestras)")

    if T < 0.5:
        print(f"  [AVISO] Fase 2 muy corta ({T:.2f} s) - resultados pueden ser poco fiables")

    sw = compute_sway_features(df2, T)
    m2 = compute_mec2_features(df2, support_side=support_side)

    S1    = compute_s_mec1(sw)
    S2    = compute_s_mec2(m2)
    Stime = float(min(T / 20.0, 1.0))
    score = 0.40 * S1 + 0.30 * S2 + 0.30 * Stime

    return dict(
        Duracion_s          = T,
        RMS_AP_mm           = sw['rms_ap_mm'],
        RMS_ML_mm           = sw['rms_ml_mm'],
        Rango_AP_mm         = sw['range_ap_mm'],
        Rango_ML_mm         = sw['range_ml_mm'],
        Area_Sway_mm2       = sw['sway_area_mm2'],
        Long_Trayectoria_mm = sw['path_length_mm'],
        Vel_CoM_mm_s        = sw['mean_com_vel_mms'],
        Ratio_Tobillo_Cadera= m2['ahr'],
        Caida_Pelvica_deg   = m2['pelvic_drop_deg'],
        Rango_ML_Tronco_deg = m2['range_ml_trunk_deg'],
        S_Mec1 = S1,
        S_Mec2 = S2,
        S_Tiempo = Stime,
        Score  = score,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows = []

    for subject in ALL_SUBJECTS:
        print(f"\n{'='*45}\n{subject}")
        group = 'healthy' if subject.startswith('h') else 'blind'
        row   = {'subject': subject, 'group': group}

        for key, trial in [('a', 'test3a'), ('b', 'test3b')]:
            res = analyze_trial(subject, trial)
            if res is not None:
                for k, v in res.items():
                    row[f'T3{key.upper()}_{k}'] = v

        # Score combinado (media de los ensayos disponibles)
        avail_scores = [row[k] for k in ('T3A_Score', 'T3B_Score') if k in row]
        row['Test3_Score'] = float(np.mean(avail_scores)) if avail_scores else np.nan

        rows.append(row)

    df_new = pd.DataFrame(rows)

    # ── Actualizar CSV global ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    if os.path.exists(RESULTS_PATH):
        df_g = pd.read_csv(RESULTS_PATH, sep=';', decimal=',')
        # Eliminar columnas anteriores de test3 si se re-ejecuta
        drop = [c for c in df_g.columns if c.startswith('T3') or c == 'Test3_Score']
        df_g = df_g.drop(columns=drop, errors='ignore')
        merge_cols = [c for c in df_new.columns if c not in ('group',)]
        df_g = df_g.merge(df_new[merge_cols], on='subject', how='outer')
        df_g['group'] = df_g['group'].fillna(
            df_g['subject'].apply(lambda s: 'healthy' if s.startswith('h') else 'blind'))
    else:
        df_g = df_new

    df_g.to_csv(RESULTS_PATH, index=False, sep=';', decimal=',')
    print(f"\nResultados guardados -> {RESULTS_PATH}")

    # ── Resumen por pantalla ──────────────────────────────────────────────────
    show = ['subject', 'group',
            'T3A_Duracion_s', 'T3A_S_Mec1', 'T3A_S_Mec2', 'T3A_S_Tiempo', 'T3A_Score',
            'T3B_Duracion_s', 'T3B_S_Mec1', 'T3B_S_Mec2', 'T3B_S_Tiempo', 'T3B_Score',
            'Test3_Score']
    show = [c for c in show if c in df_new.columns]
    print('\n' + df_new[show].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
