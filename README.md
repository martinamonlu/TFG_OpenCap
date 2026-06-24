# TFG_OpenCap
Biomechanical pipeline for postural-control assessment in visually deprived
people using OpenCap markerless motion capture.

# Desarrollo e implementación de un sistema de análisis del control postural en personas con discapacidad visual mediante visión por computador
### TFG — Grado en Ingeniería Biomédica · ETSIT-UPM · 2026
**Martina Montesdeoca Luzuriaga**

---

## Overview

This repository contains the Python pipeline developed for the Bachelor's
Thesis *"Desarrollo e implementación de un sistema de análisis del control
postural en personas con discapacidad visual mediante visión por computador"*.

The system instruments **four tasks of the Mini-BESTest** clinical balance
battery (Tests 3, 7, 11 and 14) using **OpenCap**, a smartphone-based
markerless motion-capture platform, and derives a deterministic instrumented
**score (0–2 per test, 0–12 total)** comparable to the clinician's 0–2 scale.
On top of the scores it runs a full statistical comparison between groups, a
concurrent validation against the physiotherapist, an exploratory
machine-learning block (clustering + classification) and a weight-optimization
annex.

The instrumented tests span three of the four Mini-BESTest sections
(anticipatory control, sensory orientation and dynamic gait); the total is
**not** the full Mini-BESTest score, only the sum of the instrumented items.

---

## Subjects and groups

- **40 participants** (markerless capture, 2 iPhones, 60 Hz), in collaboration
  with the ONCE.
- `h01–h20` — **control** (sighted).
- `b01–b11` — **discapacidad visual** (permanent visual disability).
- `b12–b20` — **ojos cerrados** (sighted, eyes closed).

Two configurations are analysed in parallel:

- **2 groups:** *privados de visión* (`b01–b20`) vs. *control*.
- **3 groups:** *discapacidad visual* vs. *ojos cerrados* vs. *control*.

Quality control excludes invalid OpenCap captures, so each test is analysed
with its valid subjects (missing data handled per test, no imputation); the
machine-learning block uses the 34 complete cases.

---

## Project structure
```
TFG/
├── data/                  # .mot kinematics per subject (not included)
├── results/               # Output CSVs + results/figuras/ (not included)
│   └── figuras/           # PNG figures used in the thesis
├── opencap-processing/    # OpenCap API helper (external)
└── scripts/
    ├── utils.py                   # Shared: loading, filtering, stomp detection, trim/quality helpers
    ├── 00_download_all.py         # Download .mot files from the OpenCap API
    ├── trim_all.py                # STEP 1 — segment every .mot to the real test interval (trimmed/)
    ├── 01_test3_analysis.py       # Test 3  — single-leg stance (3a / 3b)
    ├── 02_test7_analysis.py       # Test 7  — bipedal stance, feet together
    ├── 03_test11_analysis.py      # Test 11 — walking with head turns
    ├── 04_test14_analysis.py      # Test 14 — Timed Up & Go (simple 14-1 / dual-task 14-2)
    ├── 05_control_calidad.py      # Quality control (auto + manual review) -> control_calidad.csv
    ├── 06_statistical_analysis.py # Decision-tree stats + effect sizes + Spearman validation
    ├── 07_graficas.py             # All thesis figures + descriptive feature table
    ├── 08_machine_learning.py     # Clustering (KMeans/Ward) + classification (LogReg/RF)
    ├── 09_optimizacion_pesos.py   # Annex: mechanism-weight optimization (Powell + L2 + CV)
    ├── exceptions.json            # Manual corrections for wrong-foot stomps
    └── session_ids.json           # OpenCap session identifiers
```

All analysis scripts read the **trimmed** files, so `trim_all.py` must run
first.

---

## Requirements

**Python 3.13.** Dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib scikit-posthocs
```

- `scipy` — Shapiro-Wilk, Levene, t-Student, Mann-Whitney, ANOVA, Kruskal-Wallis,
  Tukey HSD, Spearman, Powell optimizer.
- `scikit-posthocs` — Dunn post-hoc test with Bonferroni correction.
- `scikit-learn` — clustering, classification and cross-validation.
- `00_download_all.py` additionally relies on the local `opencap-processing/`
  helper and the OpenCap account credentials.

---

## Pipeline (run order)

```
00_download_all.py    # (once) fetch raw .mot from OpenCap
trim_all.py           # segment every capture  -> data/<subj>/.../trimmed/
01–04_*_analysis.py   # per-test instrumented scores -> resultados_globales.csv
05_control_calidad.py # quality flags -> control_calidad.csv
06_statistical_analysis.py
07_graficas.py
08_machine_learning.py
09_optimizacion_pesos.py
```

---

## Methods summary

- **Instrumented scoring.** Deterministic, no learning: per test the
  biomechanical features (CoM medio-lateral sway, Ankle-Hip Ratio, pelvic
  drop / list, gait cadence and variability, TUG time, lateral deviation) are
  normalized to [0–1] and combined with biomechanically-fixed weights into a
  0–2 score; the six sub-tests sum to a 0–12 total.
- **Statistics (decision tree).** For each continuous variable, normality
  (Shapiro-Wilk) and homoscedasticity (Levene) route the comparison to a
  **parametric** (t-Student / one-way ANOVA + Tukey) or **non-parametric**
  (Mann-Whitney / Kruskal-Wallis + Dunn-Bonferroni) test. Clinical ordinal
  scores always go non-parametric. Effect sizes: **Cohen d, Rosenthal r,
  η², ε²**.
- **Concurrent validation.** Spearman correlation between each instrumented
  score and the physiotherapist's clinical score.
- **Machine learning (exploratory, n = 34).** Unsupervised clustering (KMeans
  and Ward, elbow method, ARI vs. real labels) and supervised classification
  (logistic regression vs. Random Forest) with **repeated stratified
  cross-validation** (5 folds × 20 repeats), reporting accuracy, precision,
  recall, F1 and AUC against a majority-class baseline.
- **Weight optimization (annex).** Powell (derivative-free) + L2, validated by
  5-fold stratified CV, to check the a-priori mechanism weights are near
  optimal without overfitting.

---

## Outputs (`results/`)

| File | Content |
|------|---------|
| `resultados_globales.csv` | Per-subject features + instrumented & clinical scores |
| `control_calidad.csv`     | Capture quality (auto + manual review) |
| `estadistica_2grupos.csv` / `estadistica_3grupos.csv` | Group comparisons (decision-tree route, p, effect size) |
| `estadistica_Spearman.csv`| Instrumented vs. clinical correlation |
| `features_descriptivas.csv`| Median/IQR per group of representative features |
| `clusters_asignacion.csv` / `clusters_resumen.csv` | Cluster labels and metrics |
| `clasificacion_metricas.csv` | Classification performance |
| `optimizacion_pesos.csv`  | Theoretical vs. optimized weights (annex) |
| `figuras/`                | All PNG figures used in the thesis |

---

## Citation

> Montesdeoca Luzuriaga, M. (2026). *Desarrollo e implementación de un sistema
> de análisis del control postural en personas con discapacidad visual
> mediante visión por computador*. Trabajo Fin de Grado, ETSIT-UPM.

---

## License

MIT License.
