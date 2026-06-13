"""
Test 14 - Timed Up and Go, simple y con doble tarea (test14-1.mot, test14-2.mot)
==================================================================================
Score_inst = 0.65 * S_Tiempo + 0.35 * S_Estabilidad

test14-1 = TUG simple
test14-2 = TUG con doble tarea cognitiva
Se analizan de forma independiente con la misma funcion.

SCORING (robusto, sin depender de la deteccion de fases):
  S_Tiempo      - tiempo total de la prueba. El TUG clinico es un test
                  cronometrado; es la feature que mejor discrimina VIP vs sano
                  (p=0.001). Umbrales good=5s, poor=12s.
  S_Estabilidad - desviacion lateral RMS de la marcha (PCA, igual que test11),
                  medida sobre todo el ensayo. Control de la trayectoria; los
                  VIP se desvian mas. Umbrales good=35mm, poor=70mm.

Por que NO se usan las fases en el score:
  La descomposicion en 5 fases (levantarse/marcha/giro/vuelta/sentarse) mediante
  pelvis_ty y pelvis_tz resulto poco fiable con la resolucion de captura de
  OpenCap: la transicion sentado/de pie y el avance de marcha apenas se
  registran en la mayoria de ensayos, por lo que la deteccion falla en ~84% de
  los casos. Las metricas dependientes de fase (transiciones, giro, velocidad)
  quedaban degeneradas, y la oscilacion de tronco/pelvis no discriminaba o lo
  hacia en direccion contraria. El score se construye solo con lo robusto y
  bien orientado: tiempo total y desviacion lateral.

Las fases se siguen detectando (detect_phases_tug) y sus tiempos se guardan
como columnas informativas, pero no entran en el score.
"""

import sys
import os
import numpy as np
import pandas as pd

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.append(r'C:\Users\marti\Desktop\TFG\scripts')
from utils import ALL_SUBJECTS, SAMPLING_RATE, lowpass_filter, load_trimmed, is_excluded

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


# ── PCA: desviación lateral (igual que test11) ────────────────────────────────

def compute_lateral_deviation_pca(tz, tx):
    """
    Desviación lateral RMS (mm) usando PCA sobre la trayectoria 2D del pelvis.
    PC1 = dirección de marcha, PC2 = dirección lateral (perpendicular).
    """
    if len(tz) < 2:
        return 0.0

    coords   = np.column_stack([tz, tx])
    coords_c = coords - coords.mean(axis=0)
    cov      = np.cov(coords_c.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    pc2 = eigenvectors[:, 0]                 # menor eigenvalue → lateral
    lateral = coords_c @ pc2
    return float(np.sqrt(np.mean(lateral**2)) * 1000.0)   # m → mm


# ── Detección de fases del TUG ────────────────────────────────────────────────

def detect_phases_tug(df, fs=FS):
    """
    Detecta los índices que delimitan las 5 fases del TUG a partir de
    pelvis_ty (sentarse/levantarse) y pelvis_tz (marcha ida/giro/vuelta).

    Devuelve un dict con idx_f1_end, idx_f2_end, idx_f3_end, idx_f5_start,
    o None si la detección falla (usar fallback de ensayo completo).
    """
    pelvis_ty = lowpass_filter(df['pelvis_ty'].values, cutoff=3.0, fs=fs)
    pelvis_tz = lowpass_filter(df['pelvis_tz'].values, cutoff=3.0, fs=fs)

    n_edge = max(1, int(0.3 * fs))
    sit_y_start = float(np.mean(pelvis_ty[:n_edge]))
    sit_y_end   = float(np.mean(pelvis_ty[-n_edge:]))
    sit_y       = (sit_y_start + sit_y_end) / 2.0

    stand_y = float(np.percentile(pelvis_ty, 90))
    rng_y   = stand_y - sit_y

    if rng_y < 0.03:   # < 3 cm → no se aprecia transición sentado/de pie
        return None

    thresh_y  = sit_y + 0.9 * rng_y
    above_idx = np.where(pelvis_ty >= thresh_y)[0]

    if len(above_idx) == 0:
        return None

    idx_f1_end   = int(above_idx[0])
    idx_f5_start = int(above_idx[-1])

    if idx_f5_start <= idx_f1_end:
        return None

    # Giro: máximo desplazamiento absoluto de pelvis_tz dentro de F2-F4
    tz_seg = pelvis_tz[idx_f1_end:idx_f5_start + 1]
    tz_rel = tz_seg - tz_seg[0]
    idx_turn_local = int(np.argmax(np.abs(tz_rel)))
    max_disp = float(np.abs(tz_rel[idx_turn_local]))

    if max_disp < 0.3:   # < 30 cm → no hay marcha de ida/vuelta clara
        return None

    # Ventana de giro: muestras donde |tz_rel| >= 90% del desplazamiento máximo
    turn_mask = np.abs(tz_rel) >= 0.9 * max_disp
    turn_idx  = np.where(turn_mask)[0] + idx_f1_end

    idx_f2_end = int(turn_idx[0])    # fin de marcha de ida / inicio del giro
    idx_f3_end = int(turn_idx[-1])   # fin del giro / inicio de marcha de vuelta

    return dict(
        idx_f1_end   = idx_f1_end,
        idx_f2_end   = idx_f2_end,
        idx_f3_end   = idx_f3_end,
        idx_f5_start = idx_f5_start,
    )


# ── Cálculo de features ───────────────────────────────────────────────────────

def compute_features(df, phases, fs=FS):
    """
    Calcula todas las features del TUG a partir del DataFrame recortado
    y las fases detectadas (o None si se usa el ensayo completo como fallback).
    """
    time        = df['time'].values
    pelvis_tz   = df['pelvis_tz'].values
    pelvis_tx   = df['pelvis_tx'].values
    pelvis_list = df['pelvis_list'].values if 'pelvis_list' in df.columns else np.zeros(len(df))
    lb          = df['lumbar_bending'].values
    hip_r       = df['hip_flexion_r'].values
    hip_l       = df['hip_flexion_l'].values
    hip_mean    = (hip_r + hip_l) / 2.0

    dt      = 1.0 / fs
    T_total = float(time[-1] - time[0])

    if phases is not None:
        idx_f1_end   = phases['idx_f1_end']
        idx_f2_end   = phases['idx_f2_end']
        idx_f3_end   = phases['idx_f3_end']
        idx_f5_start = phases['idx_f5_start']

        T_levantarse = float(time[idx_f1_end]   - time[0])
        T_sentarse   = float(time[-1]           - time[idx_f5_start])
        T_giro       = float(time[idx_f3_end]   - time[idx_f2_end])

        idx_f1   = np.arange(0, idx_f1_end + 1)
        idx_f2   = np.arange(idx_f1_end, idx_f2_end + 1)
        idx_f4   = np.arange(idx_f3_end, idx_f5_start + 1)
        idx_walk = np.concatenate([idx_f2, idx_f4])
    else:
        T_levantarse = 0.0
        T_sentarse   = 0.0
        T_giro       = np.nan
        idx_f1       = np.arange(0, max(1, int(0.1 * len(df))))
        idx_walk     = np.arange(0, len(df))

    # Velocidad de marcha (informativa). Se calcula sobre las fases de marcha
    # si se detectaron; si no, sobre todo el ensayo.
    tz_walk = pelvis_tz[idx_walk]
    if len(idx_walk) >= 2:
        dist_walk  = float(np.sum(np.abs(np.diff(tz_walk))))
        T_walk     = len(idx_walk) * dt
        vel_marcha = dist_walk / T_walk if T_walk > 0 else np.nan
        vel_inst   = np.abs(np.diff(tz_walk)) / dt
        sd_vel     = float(np.std(vel_inst))
    else:
        vel_marcha = np.nan
        sd_vel     = np.nan

    # Estabilidad de la marcha SOBRE TODO EL ENSAYO (independiente de fases).
    # La descomposicion fina por fases no es fiable con la resolucion de captura
    # (falla en ~84% de los ensayos), asi que las metricas de estabilidad y el
    # score no dependen de ella. Las fases se conservan solo para reporte.
    sd_pelvic_list = float(np.std(pelvis_list))
    dev_lat_rms_mm = compute_lateral_deviation_pca(pelvis_tz, pelvis_tx)
    trunk_ml_rms   = float(np.std(lb))

    # Flexion maxima de cadera durante el levantarse (F1) si hay fases.
    flex_cadera_max = float(np.max(hip_mean[idx_f1]))

    return dict(
        T_total_s            = T_total,
        Vel_Marcha_m_s       = vel_marcha,
        T_Giro_s             = T_giro,
        T_Levantarse_s       = T_levantarse,
        T_Sentarse_s         = T_sentarse,
        SD_Vel_Marcha_m_s    = sd_vel,
        SD_Pelvic_List_deg   = sd_pelvic_list,
        Desv_Lateral_RMS_mm  = dev_lat_rms_mm,
        Trunk_ML_RMS_deg     = trunk_ml_rms,
        FlexCadera_Max_F1_deg= flex_cadera_max,
    )


# ── Score functions ────────────────────────────────────────────────────────────
# El score del TUG se basa en dos componentes ROBUSTOS y bien orientados:
#   S_tiempo      - tiempo total de la prueba (el TUG clinico es cronometrado;
#                   es la unica feature que discrimina con fuerza, p=0.001).
#   S_estabilidad - desviacion lateral de la marcha (control de la trayectoria;
#                   VIP se desvian mas, dir. correcta).
# Se descartaron las metricas de fase (transiciones, giro, velocidad) porque la
# deteccion de fases falla en ~84% de los ensayos, y la oscilacion de tronco/
# pelvis porque no discrimina o lo hace en direccion contraria en esta muestra.
# Umbrales good/poor = p25 sano / p75 VIP (mismo criterio que en el test11).

def compute_s_tiempo(f):
    return norm_inv(f['T_total_s'], good=5.0, poor=12.0)

def compute_s_estabilidad(f):
    return norm_inv(f['Desv_Lateral_RMS_mm'], good=35.0, poor=70.0)


# ── Per-trial analysis ────────────────────────────────────────────────────────

def analyze_trial(subject, trial):
    """
    Carga, recorta y analiza un ensayo de Test 14 (test14-1 o test14-2).
    Devuelve un dict con todas las features y scores, o None si el archivo no existe.
    """
    if is_excluded(subject, trial):
        print(f"  [EXCLUIDO] {subject}/{trial}: marcado en revision manual")
        return None

    try:
        df = load_trimmed(subject, trial)
    except FileNotFoundError:
        print(f"  [SKIP] {subject}/{trial}: archivo no encontrado")
        return None
    except Exception as e:
        print(f"  [ERROR] {subject}/{trial}: {e}")
        return None

    T_total = float(df['time'].iloc[-1] - df['time'].iloc[0])
    print(f"  Duración ensayo: {T_total:.2f} s  ({len(df)} muestras)")

    phases = detect_phases_tug(df)
    if phases is None:
        print(f"  [PARCIAL] {subject}/{trial}: fases no detectadas -> usando ensayo completo")
    else:
        print(f"  Fases detectadas: F1=0-{phases['idx_f1_end']}, "
              f"F2={phases['idx_f1_end']}-{phases['idx_f2_end']}, "
              f"F3={phases['idx_f2_end']}-{phases['idx_f3_end']}, "
              f"F4={phases['idx_f3_end']}-{phases['idx_f5_start']}, "
              f"F5={phases['idx_f5_start']}-{len(df)-1}")

    f = compute_features(df, phases)

    S_tiempo = compute_s_tiempo(f)
    S_estab  = compute_s_estabilidad(f)
    score    = 0.65 * S_tiempo + 0.35 * S_estab

    return dict(
        T_Total_s             = f['T_total_s'],
        Vel_Marcha_m_s        = f['Vel_Marcha_m_s'],
        T_Giro_s              = f['T_Giro_s'],
        T_Levantarse_s        = f['T_Levantarse_s'],
        T_Sentarse_s          = f['T_Sentarse_s'],
        SD_Vel_Marcha_m_s     = f['SD_Vel_Marcha_m_s'],
        SD_Pelvic_List_deg    = f['SD_Pelvic_List_deg'],
        Desv_Lateral_RMS_mm   = f['Desv_Lateral_RMS_mm'],
        Trunk_ML_RMS_deg      = f['Trunk_ML_RMS_deg'],
        FlexCadera_Max_F1_deg = f['FlexCadera_Max_F1_deg'],
        S_Tiempo       = S_tiempo,
        S_Estabilidad  = S_estab,
        Score          = score,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rows = []

    for subject in ALL_SUBJECTS:
        print(f"\n{'='*45}\n{subject}")
        group = 'healthy' if subject.startswith('h') else 'blind'
        row   = {'subject': subject, 'group': group}

        for key, trial in [('1', 'test14-1'), ('2', 'test14-2')]:
            print(f"\n -- {trial} --")
            res = analyze_trial(subject, trial)
            if res is not None:
                for k, v in res.items():
                    row[f'T14_{key}_{k}'] = v

        rows.append(row)

    df_new = pd.DataFrame(rows)

    # ── Actualizar CSV global ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)

    if os.path.exists(RESULTS_PATH):
        df_g = pd.read_csv(RESULTS_PATH, sep=';', decimal=',')
        drop = [c for c in df_g.columns if c.startswith('T14_')]
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
            'T14_1_T_Total_s', 'T14_1_Desv_Lateral_RMS_mm',
            'T14_1_S_Tiempo', 'T14_1_S_Estabilidad', 'T14_1_Score',
            'T14_2_T_Total_s', 'T14_2_Desv_Lateral_RMS_mm',
            'T14_2_S_Tiempo', 'T14_2_S_Estabilidad', 'T14_2_Score']
    show = [c for c in show if c in df_new.columns]
    print('\n' + df_new[show].round(3).to_string(index=False))


if __name__ == '__main__':
    main()
