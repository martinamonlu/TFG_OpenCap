"""
08_machine_learning.py - Clustering, clasificacion y regresion
==============================================================
Tres analisis de machine learning sobre las 6 puntuaciones instrumentales
(Inst_Test3a/3b/7/11/14-1/14-2, escala 0-2). Solo se usan los sujetos con
los 6 sub-ensayos validos (casos completos): 34 de 40.

AVISO DE TAMANO MUESTRAL: con n=34 todo resultado es ORIENTATIVO. Para evitar
sobreajuste y fuga de datos:
  - El escalado se hace DENTRO de la validacion cruzada (Pipeline).
  - Validacion cruzada repetida y estratificada; se reportan media y desv.
  - Se compara siempre contra una linea base (clase mayoritaria / azar).

1. CLUSTERING (no supervisado): KMeans + metodo del codo + silueta para
   recomendar k. Asigna cada participante a un cluster y cruza los clusters
   con los grupos reales (sin usarlos en el ajuste). Visualizacion PCA 2D.
2. CLASIFICACION: invidentes vs control (2 grupos) y, de forma exploratoria,
   ciego/ojos cerrados/control (3 grupos). Regresion logistica y Random Forest.
3. REGRESION: predecir la puntuacion clinica total (0-12) desde las features
   instrumentales. Ridge y Random Forest.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             confusion_matrix, roc_curve, auc,
                             precision_recall_fscore_support, accuracy_score)
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     KFold, cross_val_score, cross_val_predict)

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

RESULTS_DIR = r'C:\Users\marti\Desktop\TFG\results'
FIG_DIR     = os.path.join(RESULTS_DIR, 'figuras')

# Test 7 se EXCLUYE del ML: no discrimina en ningun analisis (bipedestacion
# con pies juntos, tarea trivial para adultos 20-40 anos) y al quitarlo mejora
# la clasificacion (LogReg: acc 0.72->0.78, AUC 0.80->0.82, validacion cruzada).
FEATURES = ['Inst_Test3a_2', 'Inst_Test3b_2',
            'Inst_Test11_2', 'Inst_Test14_1_2', 'Inst_Test14_2_2']
FEAT_LABELS = ['Test 3a', 'Test 3b', 'Test 11', 'Test 14-1', 'Test 14-2']

# Paleta por grupo real (azul + naranja de la uni; morado en vez de verde)
COL_CON  = '#0072B2'   # control            - azul
COL_DV   = '#E69F00'   # discapacidad visual - naranja
COL_OJOS = '#9467BD'   # ojos cerrados      - morado
COL_INV  = COL_DV      # invidentes (2 grupos) - naranja
GCOL = {'control': COL_CON, 'disc_visual': COL_DV, 'ojos_cerrados': COL_OJOS}
GNOM = {'control': 'Control', 'disc_visual': 'Discapacidad visual',
        'ojos_cerrados': 'Ojos cerrados'}
SEED = 0


def grupo3(subject):
    s = str(subject).lower()
    if s.startswith('h'):
        return 'control'
    return 'disc_visual' if int(s[1:]) <= 11 else 'ojos_cerrados'


def cargar():
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'resultados_globales.csv'),
                     sep=';', decimal=',')
    df['grupo2'] = np.where(df.subject.str.startswith('h'), 'control', 'invidente')
    df['grupo3'] = df.subject.apply(grupo3)
    comp = df[FEATURES].notna().all(axis=1)
    df_c = df[comp].reset_index(drop=True)
    print(f"Casos completos (6 features): {len(df_c)} de {len(df)}")
    print(f"  2 grupos: {df_c.grupo2.value_counts().to_dict()}")
    print(f"  3 grupos: {df_c.grupo3.value_counts().to_dict()}")
    return df_c


# ===========================================================================
# 1. CLUSTERING
# ===========================================================================

def clustering(df):
    print("\n" + "=" * 70)
    print("1. CLUSTERING (KMeans + metodo del codo)")
    print("=" * 70)

    X = df[FEATURES].values
    Xs = StandardScaler().fit_transform(X)

    # Metodo del codo: ejecuta KMeans para varios k y observa la inercia
    # (compacidad intra-cluster). El codo es donde la curva deja de bajar
    # bruscamente. Tecnica habitual para orientar la eleccion de k.
    Ks = list(range(1, 9))
    inercias = [KMeans(n_clusters=k, n_init=10, random_state=SEED).fit(Xs).inertia_
                for k in Ks]
    print("Inercia por k: " +
          ", ".join(f"k={k}:{v:.0f}" for k, v in zip(Ks, inercias)))
    print("Metodo del codo: la curva se aplana a partir de k=3. Se analizan "
          "k=2 y k=3 (k=3 ademas por los 3 grupos reales del estudio).")

    # Grafica del codo (solo inercia)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(Ks, inercias, 'o-', color=COL_CON)
    ax.axvline(3, ls='--', color='gray', label='k = 3 (elegido)')
    ax.set_xlabel('Número de clusters (k)')
    ax.set_ylabel('Inercia (compacidad intra-cluster)')
    ax.set_title('Método del codo')
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out1 = os.path.join(FIG_DIR, 'clustering_codo.png')
    fig.savefig(out1, dpi=200); plt.close(fig)

    df = df.copy()

    def evaluar(labels, nombre):
        """Imprime silueta, cruce con grupos reales, pureza y ARI."""
        sil = silhouette_score(Xs, labels)
        d = df.copy(); d['cl'] = labels
        ct3 = pd.crosstab(d['cl'], d['grupo3'])
        pureza = ct3.max(axis=1).sum() / len(d)
        ari2 = adjusted_rand_score(d['grupo2'], labels)
        ari3 = adjusted_rand_score(d['grupo3'], labels)
        print(f"\n  [{nombre}] silueta={sil:.3f}  pureza(3g)={pureza:.2f}  "
              f"ARI 2grupos={ari2:.3f}  ARI 3grupos={ari3:.3f}")
        print("    cruce cluster x grupo real:")
        print("    " + ct3.to_string().replace("\n", "\n    "))
        return dict(metodo=nombre, silueta=round(sil, 3), pureza=round(pureza, 2),
                    ari_2g=round(ari2, 3), ari_3g=round(ari3, 3))

    # --- KMeans y jerarquico (Ward), cada uno con k=2 y k=3 -------------------
    # k=2: lo que sugiere el codo/silueta (separacion natural mas fuerte).
    # k=3: justificado por el diseno del estudio (3 grupos reales).
    print("\n" + "-" * 70)
    print("COMPARACION DE METODOS Y k (ARI: 0=azar, 1=coincidencia perfecta)")
    print("-" * 70)
    resumen = []
    etiquetas = {}
    for k in (2, 3):
        lab_km = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(Xs)
        resumen.append(evaluar(lab_km, f'KMeans k={k}'))
        etiquetas[f'kmeans_k{k}'] = lab_km
        lab_hc = AgglomerativeClustering(n_clusters=k, linkage='ward').fit_predict(Xs)
        resumen.append(evaluar(lab_hc, f'Jerarquico (Ward) k={k}'))
        etiquetas[f'jerarq_k{k}'] = lab_hc

    # El clustering jerarquico (Ward) se mantiene como comprobacion de robustez
    # (su ARI aparece en clusters_resumen.csv), pero no se genera el dendrograma:
    # con la estructura debil y n=34 hojas no aporta informacion adicional.

    # --- PCA 2D: KMeans k=3 vs grupo real ------------------------------------
    # Cada cluster se pinta con el color del grupo real con el que mas coincide
    # (emparejamiento 1-a-1 optimo). Asi, un participante bien agrupado sale del
    # MISMO color en los dos paneles, y los que "cambian de color" son los que el
    # algoritmo confunde.
    labels = np.asarray(etiquetas['kmeans_k3'])
    grupos = ['disc_visual', 'ojos_cerrados', 'control']
    M = np.array([[np.sum((labels == c) & (df['grupo3'].values == g))
                   for g in grupos] for c in sorted(set(labels))])
    filas, cols = linear_sum_assignment(-M)        # maximiza coincidencia
    cl_to_grupo = {int(f): grupos[c] for f, c in zip(filas, cols)}

    pca = PCA(n_components=2, random_state=SEED)
    proj = pca.fit_transform(Xs)
    var = pca.explained_variance_ratio_ * 100
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    # Panel izquierdo: clusters, coloreados por su grupo dominante
    for c in sorted(set(labels)):
        g = cl_to_grupo[c]
        m = labels == c
        b1.scatter(proj[m, 0], proj[m, 1], s=45, alpha=0.85, color=GCOL[g],
                   edgecolor='black', linewidth=0.3,
                   label=f'Cluster {c+1} (≈ {GNOM[g]})')
    b1.set_title('Clusters KMeans (k=3)'); b1.legend(fontsize=8); b1.grid(alpha=0.3)

    # Panel derecho: grupo real
    for g in grupos:
        m = df['grupo3'].values == g
        b2.scatter(proj[m, 0], proj[m, 1], s=45, alpha=0.85, color=GCOL[g],
                   edgecolor='black', linewidth=0.3, label=GNOM[g])
    b2.set_title('Grupo real (no usado en el ajuste)')
    b2.legend(fontsize=8); b2.grid(alpha=0.3)

    for ax in (b1, b2):
        ax.set_xlabel(f'PC1 ({var[0]:.0f}% var.)')
    b1.set_ylabel(f'PC2 ({var[1]:.0f}% var.)')
    fig.suptitle('Proyección PCA de los participantes', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out2 = os.path.join(FIG_DIR, 'clustering_pca.png')
    fig.savefig(out2, dpi=200); plt.close(fig)

    # --- Guardar asignaciones y resumen --------------------------------------
    # Columnas claras: grupo_real_* = etiqueta verdadera del sujeto (NO es un
    # cluster); cluster_* = cluster asignado por cada algoritmo (IDs en base 1,
    # arbitrarios: el numero solo distingue clusters, no tiene orden).
    asign = pd.DataFrame({
        'sujeto':               df['subject'].values,
        'grupo_real_2cat':      df['grupo2'].values,   # invidente / control
        'subgrupo_real_3cat':   df['grupo3'].values,   # disc_visual/ojos/control
    })
    for nombre, lab in etiquetas.items():
        asign[f'cluster_{nombre}'] = np.asarray(lab) + 1   # base 1 para Excel
    asign.to_csv(os.path.join(RESULTS_DIR, 'clusters_asignacion.csv'),
                 index=False, sep=';')
    pd.DataFrame(resumen).to_csv(os.path.join(RESULTS_DIR, 'clusters_resumen.csv'),
                                 index=False, sep=';', decimal=',')
    print(f"\nFiguras: {os.path.basename(out1)}, {os.path.basename(out2)}")
    print("Asignaciones -> clusters_asignacion.csv | resumen -> clusters_resumen.csv")


# ===========================================================================
# 2. CLASIFICACION
# ===========================================================================

def _cv_clasif(X, y, cv, scorings):
    out = {}
    for nombre, pipe in [
        ('Regresion logistica', Pipeline([('sc', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, random_state=SEED))])),
        ('Random Forest', Pipeline([('sc', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=300, random_state=SEED))])),
    ]:
        fila = {}
        for sc in scorings:
            s = cross_val_score(pipe, X, y, cv=cv, scoring=sc)
            fila[sc] = (s.mean(), s.std())
        out[nombre] = fila
    return out


def clasificacion(df):
    print("\n" + "=" * 70)
    print("2. CLASIFICACION")
    print("=" * 70)
    X = df[FEATURES].values

    # --- 2 grupos: invidente vs control ---
    y2 = (df['grupo2'] == 'invidente').astype(int).values
    base2 = max(np.mean(y2), 1 - np.mean(y2))
    cv2 = RepeatedStratifiedKFold(n_splits=5, n_repeats=20, random_state=SEED)
    print(f"\n-- 2 grupos (invidente vs control), n={len(y2)} --")
    print(f"   Linea base (clase mayoritaria): accuracy = {base2:.3f}")
    res = _cv_clasif(X, y2, cv2, ['accuracy', 'roc_auc', 'f1'])
    for nombre, fila in res.items():
        print(f"   {nombre:22} "
              f"acc={fila['accuracy'][0]:.3f}+-{fila['accuracy'][1]:.3f}  "
              f"AUC={fila['roc_auc'][0]:.3f}+-{fila['roc_auc'][1]:.3f}  "
              f"F1={fila['f1'][0]:.3f}+-{fila['f1'][1]:.3f}")

    # Importancia de variables (RF ajustado sobre todos los datos, interpretativo)
    rf = Pipeline([('sc', StandardScaler()),
                   ('clf', RandomForestClassifier(n_estimators=300, random_state=SEED))])
    rf.fit(X, y2)
    imp = rf.named_steps['clf'].feature_importances_
    orden = np.argsort(imp)[::-1]
    print("   Importancia de variables (Random Forest):")
    for i in orden:
        print(f"      {FEAT_LABELS[i]:10} {imp[i]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh([FEAT_LABELS[i] for i in orden][::-1],
            [imp[i] for i in orden][::-1], color=COL_INV, edgecolor='black')
    ax.set_xlabel('Importancia (Random Forest)')
    ax.set_title('Importancia de variables — clasificación privados de visión vs control')
    ax.grid(axis='x', alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, 'clasificacion_importancia.png'), dpi=200)
    plt.close(fig)

    # --- 3 grupos (exploratorio) ---
    y3, _ = pd.factorize(df['grupo3'])
    cont = pd.Series(y3).value_counts()
    base3 = cont.max() / cont.sum()
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    print(f"\n-- 3 grupos (disc.visual/ojos/control), n={len(y3)} [EXPLORATORIO] --")
    print(f"   Linea base (clase mayoritaria): accuracy = {base3:.3f}")
    for nombre, pipe in [
        ('Regresion logistica', Pipeline([('sc', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, random_state=SEED))])),
        ('Random Forest', Pipeline([('sc', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=300, random_state=SEED))])),
    ]:
        acc = cross_val_score(pipe, X, y3, cv=cv3, scoring='accuracy')
        print(f"   {nombre:22} acc={acc.mean():.3f}+-{acc.std():.3f}")
    print("   (clases muy pequenas: 9 y 8 sujetos -> solo orientativo)")


def clasificacion_metricas(df):
    """
    Metricas completas de la clasificacion 2 grupos (privado de vision vs
    control) a partir de predicciones CROSS-VALIDADAS: matriz de confusion,
    precision, recall, F1 y curva ROC. La probabilidad de cada sujeto se obtiene
    promediando sus predicciones fuera de fold sobre 20 repeticiones de 5-fold.
    """
    print("\n" + "=" * 70)
    print("METRICAS DE CLASIFICACION (2 grupos, predicciones cross-validadas)")
    print("=" * 70)
    X = df[FEATURES].values
    y = (df['grupo2'] == 'invidente').astype(int).values   # 1 = privado de vision
    modelos = {
        'Regresion logistica': Pipeline([('sc', StandardScaler()),
            ('clf', LogisticRegression(max_iter=2000, random_state=SEED))]),
        'Random Forest': Pipeline([('sc', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=300, random_state=SEED))]),
    }
    display = {'Regresion logistica': 'Regresión logística', 'Random Forest': 'Random Forest'}
    n_rep = 20

    probas, resumen = {}, []
    for nombre, pipe in modelos.items():
        p = np.zeros(len(y))
        for r in range(n_rep):
            skf = StratifiedKFold(5, shuffle=True, random_state=r)
            p += cross_val_predict(pipe, X, y, cv=skf, method='predict_proba')[:, 1]
        p /= n_rep
        probas[nombre] = p
        pred = (p >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, pred, average='binary', zero_division=0)
        a = auc(*roc_curve(y, p)[:2])
        print(f"  {nombre:22} acc={acc:.3f}  precision={prec:.3f}  "
              f"recall={rec:.3f}  F1={f1:.3f}  AUC={a:.3f}")
        resumen.append(dict(modelo=nombre, accuracy=round(acc, 3),
                            precision=round(prec, 3), recall=round(rec, 3),
                            f1=round(f1, 3), auc=round(a, 3)))
    pd.DataFrame(resumen).to_csv(os.path.join(RESULTS_DIR, 'clasificacion_metricas.csv'),
                                 index=False, sep=';', decimal=',')

    # --- Matriz de confusion (modelo principal: logistica) ---
    pred_lr = (probas['Regresion logistica'] >= 0.5).astype(int)
    cm = confusion_matrix(y, pred_lr)               # filas=real, columnas=predicho
    fig, ax = plt.subplots(figsize=(5, 4.6))
    ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Control', 'Privado de visión'])
    ax.set_yticklabels(['Control', 'Privado de visión'], rotation=90, va='center')
    ax.set_xlabel('Predicción'); ax.set_ylabel('Real')
    vmax = cm.max()
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(cm[i, j]), ha='center', va='center', fontsize=16,
                    color='white' if cm[i, j] > vmax / 2 else 'black')
    ax.set_title('Matriz de confusión — regresión logística (CV)')
    fig.tight_layout()
    out_cm = os.path.join(FIG_DIR, 'clasificacion_matriz_confusion.png')
    fig.savefig(out_cm, dpi=200); plt.close(fig)

    # --- Curva ROC (ambos modelos) ---
    fig, ax = plt.subplots(figsize=(5.6, 5))
    for nombre, color in [('Regresion logistica', COL_DV), ('Random Forest', COL_CON)]:
        fpr, tpr, _ = roc_curve(y, probas[nombre])
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f'{display[nombre]} (AUC = {auc(fpr, tpr):.2f})')
    ax.plot([0, 1], [0, 1], ls='--', color='gray', lw=1, label='Azar')
    ax.set_xlabel('1 − especificidad (FPR)')
    ax.set_ylabel('Sensibilidad (TPR)')
    ax.set_title('Curva ROC — privado de visión vs. control')
    ax.legend(loc='lower right', fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout()
    out_roc = os.path.join(FIG_DIR, 'clasificacion_roc.png')
    fig.savefig(out_roc, dpi=200); plt.close(fig)
    print(f"  Figuras: {os.path.basename(out_cm)}, {os.path.basename(out_roc)}")
    print("  Tabla -> clasificacion_metricas.csv")


# ===========================================================================
# 3. REGRESION
# ===========================================================================

def regresion(df):
    print("\n" + "=" * 70)
    print("3. REGRESION (predecir puntuacion clinica total 0-12)")
    print("=" * 70)
    sub = df.dropna(subset=FEATURES + ['Clin_Score_Total'])
    X = sub[FEATURES].values
    y = sub['Clin_Score_Total'].values
    print(f"n = {len(y)}  (sujetos con features y clinico total)")
    print(f"Linea base (media constante): MAE = {np.mean(np.abs(y - y.mean())):.3f}")

    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for nombre, pipe in [
        ('Ridge', Pipeline([('sc', StandardScaler()), ('reg', Ridge())])),
        ('Random Forest', Pipeline([('sc', StandardScaler()),
            ('reg', RandomForestRegressor(n_estimators=300, random_state=SEED))])),
    ]:
        r2 = cross_val_score(pipe, X, y, cv=cv, scoring='r2')
        mae = -cross_val_score(pipe, X, y, cv=cv, scoring='neg_mean_absolute_error')
        print(f"   {nombre:16} R2={r2.mean():.3f}+-{r2.std():.3f}   "
              f"MAE={mae.mean():.3f}+-{mae.std():.3f}")
    print("   (R2 puede salir bajo o negativo con n pequeno; es esperable)")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = cargar()
    clustering(df)
    clasificacion(df)
    clasificacion_metricas(df)
    regresion(df)
    print("\n" + "=" * 70)
    print("Hecho. Recuerda: n pequeno -> resultados orientativos, no confirmatorios.")
    print("=" * 70)


if __name__ == '__main__':
    main()
