"""
00_download_all.py — Descarga automática de .mot desde OpenCap
==============================================================
Implementación directa contra la API REST de OpenCap.
No requiere instalar opencap-processing ni OpenSim.

ÚNICO REQUISITO:
    pip install requests

CÓMO USARLO:
    1. Rellena session_ids.json con los IDs de cada sesión
       (cópialos de la URL: app.opencap.ai/session/<session_id>)
    2. python3 scripts/00_download_all.py
    3. Introduce tu usuario y contraseña de OpenCap cuando los pida.
       Las credenciales se guardan en .env para la próxima vez.
"""

import json
import os
import sys
import getpass
import urllib.request
import shutil

try:
    import requests
except ImportError:
    print("[ERROR] Falta 'requests'. Instálalo con:  pip install requests")
    sys.exit(1)

# ── Configuración ─────────────────────────────────────────────────────────────

API_URL          = 'https://api.opencap.ai/'
SESSION_IDS_PATH = os.path.join(os.path.dirname(__file__), 'session_ids.json')
ENV_PATH         = os.path.join(os.path.dirname(__file__), '.env')
DATA_ROOT        = r'C:\Users\marti\Desktop\TFG\data'

# Tests a descargar. None para descargar todos los trials.
# El script busca coincidencias ignorando guiones (test3a == test3-a)
# y guarda cada archivo con el nombre especificado en TRIALS.
# Así, test3a/test3b se guardan sin guion, mientras que test14-1/test14-2 se guardan con guion.
TRIALS = ['test3a', 'test3b', 'test7', 'test11', 'test14-1', 'test14-2']

# ── Autenticación ─────────────────────────────────────────────────────────────

def load_token_from_env():
    """Lee el token guardado en .env si existe."""
    if not os.path.exists(ENV_PATH):
        return None
    with open(ENV_PATH) as f:
        for line in f:
            if line.startswith('API_TOKEN='):
                return line.strip().split('=', 1)[1]
    return None

def save_token_to_env(token):
    """Guarda el token en .env para no pedir credenciales cada vez."""
    with open(ENV_PATH, 'w') as f:
        f.write(f'API_TOKEN={token}\n')
    print(f"  Token guardado en {ENV_PATH}")

def get_token():
    """Devuelve el token de autenticación (lo pide si no está guardado)."""
    token = load_token_from_env()
    if token:
        print("  Usando token guardado en .env")
        return token

    print("\nIntroduce tus credenciales de app.opencap.ai:")
    username = input("  Usuario (email): ").strip()
    password = getpass.getpass("  Contraseña: ")

    resp = requests.post(API_URL + 'login/', data={'username': username, 'password': password})

    if resp.status_code != 200:
        print(f"[ERROR] Login fallido (status {resp.status_code}). Comprueba usuario y contraseña.")
        sys.exit(1)

    token = resp.json().get('token')
    if not token:
        print("[ERROR] No se recibió token en la respuesta.")
        sys.exit(1)

    save_token_to_env(token)
    return token

# ── Llamadas a la API ─────────────────────────────────────────────────────────

def get_session(session_id, headers):
    resp = requests.get(API_URL + f'sessions/{session_id}/', headers=headers)
    if resp.status_code == 500:
        raise Exception(f'Sin respuesta del servidor. ¿Session ID correcto?')
    data = resp.json()
    if 'trials' not in data:
        raise Exception('Sesión no encontrada o sin permisos de acceso.')
    # Ordenar por fecha de creación
    data['trials'].sort(key=lambda t: t.get('created_at', ''))
    return data

def get_trial(trial_id, headers):
    return requests.get(API_URL + f'trials/{trial_id}/', headers=headers).json()

def download_file(url, dest_path):
    """Descarga un archivo desde una URL y lo guarda en dest_path."""
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()
    resp.raw.decode_content = True
    with open(dest_path, 'wb') as out:
        shutil.copyfileobj(resp.raw, out)

# ── Descarga de un sujeto ─────────────────────────────────────────────────────

def normalize(name):
    """Elimina guiones para comparar nombres: 'test3-a' == 'test3a'."""
    return name.replace('-', '')

def download_subject(subject, session_id, headers):
    """
    Descarga los .mot de un sujeto y los guarda en:
        data/<subject>/OpenSimData/Kinematics/<trial>.mot
    Los archivos se guardan siempre SIN guiones (test3a.mot, test3b.mot).
    La búsqueda en OpenCap ignora guiones (test3a encuentra test3-a y viceversa).
    """
    session = get_session(session_id, headers)

    # Índice de trials disponibles normalizado: {'test3a': trial_info, ...}
    available_norm = {}
    for t in session['trials']:
        if t['name'] in ('calibration', 'neutral'):
            continue
        available_norm[normalize(t['name'])] = t

    # Lista de trials a buscar
    if TRIALS:
        targets = list(TRIALS)
    else:
        targets = [t['name'] for t in session['trials'] if t['name'] not in ('calibration', 'neutral')]

    # Avisar si algún trial pedido no está
    for t in targets:
        if normalize(t) not in available_norm:
            print(f"  [AVISO] '{t}' no encontrado en la sesión — se salta")

    ik_folder = os.path.join(DATA_ROOT, subject, 'OpenSimData', 'Kinematics')
    os.makedirs(ik_folder, exist_ok=True)

    descargados = []
    for target in targets:
        norm_target = normalize(target)
        if norm_target not in available_norm:
            continue

        trial_info = available_norm[norm_target]
        trial_data = get_trial(trial_info['id'], headers)
        result_tags = {r['tag']: r for r in trial_data.get('results', [])}

        if 'ik_results' not in result_tags:
            print(f"  [AVISO] '{target}' no tiene ik_results todavía — ¿procesado?")
            continue

        dest = os.path.join(ik_folder, f'{target}.mot')
        if os.path.exists(dest):
            print(f"  [SKIP]  {target}.mot ya existe")
            descargados.append(target)
            continue

        url = result_tags['ik_results']['media']
        try:
            download_file(url, dest)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"  [ERROR] No se pudo descargar '{target}': {e}")
            continue
        print(f"  ✓ {target}.mot")
        descargados.append(target)

    return descargados

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # Cargar session_ids.json
    with open(SESSION_IDS_PATH, 'r') as f:
        session_ids = json.load(f)

    # Filtrar placeholders
    pendientes = {s for s, sid in session_ids.items() if sid.startswith('PEGA-AQUI')}
    if pendientes:
        print(f"\n[AVISO] {len(pendientes)} sujeto(s) sin session_id — se saltarán:")
        for s in sorted(pendientes):
            print(f"  · {s}")

    session_ids = {s: sid for s, sid in session_ids.items()
                   if not sid.startswith('PEGA-AQUI')}

    if not session_ids:
        print("[ERROR] No hay session_ids válidos. Rellena session_ids.json primero.")
        sys.exit(1)

    # Autenticar
    print(f"\n{'='*55}")
    print("  Autenticación OpenCap")
    print(f"{'='*55}")
    token   = get_token()
    headers = {'Authorization': f'Token {token}'}

    # Descargar
    print(f"\n{'='*55}")
    print(f"  Descargando {len(session_ids)} sujeto(s)")
    print(f"  Trials: {TRIALS if TRIALS else 'TODOS'}")
    print(f"{'='*55}")

    ok, errores = [], []

    for subject, session_id in session_ids.items():
        print(f"\n── {subject}  ({session_id[:8]}...)")
        try:
            descargados = download_subject(subject, session_id, headers)
            print(f"  → {len(descargados)} trial(s) listos")
            ok.append(subject)
        except Exception as e:
            print(f"  [ERROR] {e}")
            errores.append((subject, str(e)))

    # Resumen
    print(f"\n{'='*55}")
    print(f"  RESUMEN")
    print(f"{'='*55}")
    print(f"  ✓ Completados : {len(ok)}")
    print(f"  ✗ Con errores : {len(errores)}")
    if errores:
        print("\n  Sujetos con error:")
        for s, e in errores:
            print(f"    · {s}: {e}")
    print(f"\n  Archivos en: {DATA_ROOT}")
    print(f"{'='*55}\n")

if __name__ == '__main__':
    main()
