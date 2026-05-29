# TFG_OpenCap
Biomechanical analysis pipeline for postural control assessment in visually impaired people using OpenCap markerless motion capture.

# Desarrollo e implementación de un sistema de análisis del control postural en personas con discapacidad visual mediante visión por computador
### TFG — Grado en Ingeniería Biomédica · ETSIT-UPM · 2026
**Martina Montesdeoca Luzuriaga**

---

## Overview

This repository contains the Python pipeline developed for the 
Bachelor's Thesis *"Desarrollo e implementación de un sistema de 
análisis del control postural en personas con discapacidad visual 
mediante visión por computador"*.

The system instruments three tasks of the **Mini-BESTest** clinical 
balance assessment (Tests 3, 7 and 11) using **OpenCap**, a 
smartphone-based markerless motion capture platform, and computes 
continuous biomechanical scores comparable to the clinical 0–2 scale.

---

## Project Structure
TFG/
├── data/           # .mot kinematics files per subject (h01-h20, b01-b20)
├── results/        # Output CSVs
└── scripts/
├── utils.py                  # Shared functions (loading, filtering, stomp detection)
├── 00_download_all.py        # Automatic download from OpenCap API
├── 01_test3_analysis.py      # Test 3: Single-leg stance
├── 02_test7_analysis.py      # Test 7: Static bipedal stance, feet together
├── 03_test11_analysis.py     # Test 11: Walking with head turns
└── 04_statistical_analysis.py # Mann-Whitney U, effect sizes

---

## Requirements

Python 3.13.13. Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## Methods Summary

- **Motion capture:** OpenCap (2 iPhones, markerless, 60 Hz)
- **Subjects:** 40 participants — 20 healthy controls (h01–h20), 
  20 visually impaired (b01–b20), in collaboration with ONCE
- **Clinical evaluation:** Mini-BESTest scores (0–2 per item) 
  assessed by a physiotherapist
- **Biomechanical scoring:** Continuous score in [0–2] per test 
  based on CoM sway, Ankle-Hip Ratio, pelvic drop and gait parameters
- **Statistics:** Mann-Whitney U test (VIP vs. controls), 
  rank-biserial correlation as effect size

---

## Citation

If you use this code, please cite the associated thesis:

> Montesdeoca Luzuriaga, M. (2026). *Desarrollo e implementación 
> de un sistema de análisis del control postural en personas con 
> discapacidad visual mediante visión por computador*. 
> Bachelor's Thesis, ETSIT-UPM.

---

## License

This project is licensed under the MIT License.
