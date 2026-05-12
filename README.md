# 🛰️ Project 1 — GPS Satellite XYZ Positions

> **ICD-GPS-200 Algorithm · ECEF Coordinates · 3D Orbit Visualisation · Orbital Radius | 2026-01-01**

---

## 📌 Overview

A GPS receiver cannot know where the satellites are without computing their positions
from the **broadcast ephemeris**, the orbital parameters each satellite transmits
in its navigation message. This project implements the complete position computation
algorithm from scratch, producing ECEF X, Y, Z coordinates for all 32 GPS satellites
every 15 minutes over a full day.

---

## 📐 The Algorithm — ICD-GPS-200

The GPS Interface Control Document defines a 9-step procedure:

```
1. tk  = t − toe                         time from reference epoch
2. Mk  = M0 + (√(μ/a³) + Δn) · tk       mean anomaly
3. Ek  = Mk + e·sin(Ek)                  eccentric anomaly (iterate)
4. νk  = atan2(√(1−e²)·sin(Ek), cos(Ek)−e)  true anomaly
5. Φk  = νk + ω                          argument of latitude
6. uk, rk, ik  ← harmonic corrections   (Cuc, Cus, Crc, Crs, Cic, Cis)
7. xk' = rk·cos(uk),  yk' = rk·sin(uk)  orbital plane position
8. Ωk  = Ω0 + (Ωdot − ΩE)·tk − ΩE·toe  corrected ascending node
9. X, Y, Z  ← ECEF rotation             final coordinates
```

### GPS physical constants used:
```
μ  = 3.986005 × 10¹⁴ m³/s²    Earth gravitational constant
ΩE = 7.2921151467 × 10⁻⁵ rad/s Earth rotation rate
a  ≈ 26,560 km                  GPS semi-major axis
e  ≈ 0.01                       GPS eccentricity (nearly circular)
i  ≈ 55°                        GPS orbital inclination
```

---

## 🖼️ Output Plots

### Plot 1 — 3D GPS Constellation
All 32 GPS satellites drawn as coloured orbit curves around a wireframe Earth.
Dots mark each satellite's position at 00:00 UTC.

### Plot 2 — XYZ Time Series
X, Y, Z coordinates for 6 selected satellites over 24 hours.
All three components oscillate because the ECEF frame rotates with the Earth.

### Plot 3 — Orbital Radius Verification
`R = √(X²+Y²+Z²)` for all satellites — should cluster at ~26,560 km.
Small variation (~10–30 km) is due to orbital eccentricity.
This is a built-in sanity check on the computation.

---

## 📂 File Structure

```
gnss-satellite-position-computation/
│
├── output/                                   		 ← Generated plots
│   ├── plot1_3d_orbits.png
│   ├── plot2_xyz_timeseries.png
│   └── plot3_orbital_radius.png
├── src/
│   └── project1_gps_satellite_xyz_positions.py   	← Main python
├── LICENSE
├── README.md
└── requirements.txt
```

---

## ⚙️ How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set your navigation file path
In **Step 2** of the notebook:
```python
nav_path = "../../data/brdc0010.26n"   # ← update if needed
```

### 3. Run all cells
```bash
jupyter notebook src/project1_satellite_positions.ipynb
```

---

## 🛠️ Dependencies

| Package | Purpose |
|---------|---------|
| `georinex` | Parse RINEX 2/3 navigation files |
| `numpy` | Numerical computations |
| `pandas` | Time series and epoch management |
| `matplotlib` | 2D and 3D publication-quality plotting |

---

## 📡 Navigation File

```
brdc0010.26n   — RINEX 2 GPS Navigation
Date           : 2026-01-01 (Day of Year 001)
Source         : IGS CDDIS broadcast combined ephemeris
Satellites     : GPS G01–G32
Records        : ~400 ephemeris sets
```

Each satellite broadcasts a new ephemeris every **2 hours**.
The file contains one record per satellite per 2-hour validity window.

---

## 👤 Author

**Hakim El Azzouzi**
MSc Global Navigation Satellite Systems
Mohammed First University, Oujda, Morocco
📧 elazzouzihakim10@gmail.com
🔗 [linkedin.com/in/Hakim-El-Azzouzi](https://linkedin.com/in/Hakim-El-Azzouzi)
📍 Luxembourg 🇱🇺

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🔗 GNSS Navigation RINEX Series

| # | Project |
|---|---------|
| **1** | **GPS Satellite XYZ Positions** ← You are here |
| 2 | Elevation & Azimuth Angles — Sky Plot |
| 3 | Data Pre-processing — Elevation Mask · Cycle Slips · Multipath |
| 4 | Single-Point Positioning — Least Squares |
