=============================================================================
🛰️ Project 1 — GPS Satellite XYZ Positions
=============================================================================
 Author   : Hakim El Azzouzi
 Degree   : MSc Global Navigation Satellite Systems
            Mohammed First University, Oujda, Morocco
 Email    : elazzouzihakim10@gmail.com
 LinkedIn : https://linkedin.com/in/Hakim-El-Azzouzi
 Location : Luxembourg 🇱🇺
-----------------------------------------------------------------------------
 File type    : RINEX 2 GPS Navigation
 Date         : 2026-01-01 (Day of Year 001)
 Satellites   : GPS only (G01–G32)
 Records      : ~400 ephemeris sets (each valid ~2 hours)
 Source       : IGS CDDIS broadcast combined file
-----------------------------------------------------------------------------
 Description
 -----------
A GPS receiver does not know where the satellites are by default — it has to **compute their positions** from the orbital parameters that each satellite broadcasts in its navigation message.

These parameters are stored in a **RINEX navigation file** (`.26n` = RINEX 2, year 2026). This project reads those parameters and computes the **ECEF (Earth-Centred Earth-Fixed) X, Y, Z coordinates** of every GPS satellite at every 15-minute epoch across 24 hours.

ECEF is a coordinate system fixed to the Earth:
- **X-axis** → points to 0° latitude, 0° longitude (Gulf of Guinea)
- **Y-axis** → points to 0° latitude, 90° East
- **Z-axis** → points to the North Pole
- Origin = Earth's centre of mass

**3 plots produced:**

| Plot | What It Shows |
|------|---------------|
| 🌍 **3D orbit** | All GPS satellites drawn as curves around a wireframe Earth |
| 📈 **XYZ time series** | How X, Y, Z coordinates evolve over 24 hours |
| 📊 **Orbital radius** | Distance from Earth's centre — should be ~26,560 km for GPS |

---
## 📐 The GPS Orbit Algorithm (ICD-GPS-200)

Each satellite broadcasts **16 Keplerian parameters** in its navigation message.
The GPS Interface Control Document (ICD-GPS-200) defines the exact algorithm to convert these into XYZ:

```
Step 1 — Compute time from ephemeris reference epoch
         tk = t − toe

Step 2 — Compute mean anomaly
         Mk = M0 + (√(μ/a³) + Δn) · tk

Step 3 — Solve Kepler's equation (iterative) for eccentric anomaly Ek
         Ek = Mk + e · sin(Ek)   [iterate until convergence]

Step 4 — Compute true anomaly
         νk = atan2(√(1−e²)·sin(Ek), cos(Ek)−e)

Step 5 — Compute argument of latitude
         Φk = νk + ω

Step 6 — Apply second-order harmonic corrections
         uk = Φk + Cus·sin(2Φk) + Cuc·cos(2Φk)
         rk = a·(1−e·cos(Ek)) + Crs·sin(2Φk) + Crc·cos(2Φk)
         ik = i0 + IDOT·tk + Cis·sin(2Φk) + Cic·cos(2Φk)

Step 7 — Compute position in orbital plane
         xk' = rk · cos(uk)
         yk' = rk · sin(uk)

Step 8 — Compute corrected longitude of ascending node
         Ωk = Ω0 + (Ωdot − ΩE)·tk − ΩE·toe

Step 9 — Compute ECEF coordinates
         X = xk'·cos(Ωk) − yk'·cos(ik)·sin(Ωk)
         Y = xk'·sin(Ωk) + yk'·cos(ik)·cos(Ωk)
         Z = yk' · sin(ik)
```

Where:
- `a = sqrtA²` = semi-major axis (~26,560 km for GPS)
- `e` = eccentricity (~0.01 for GPS, nearly circular)
- `M0` = mean anomaly at reference epoch
- `ω` = argument of perigee
- `i0` = inclination at reference epoch (~55° for GPS)
- `Ω0` = longitude of ascending node at weekly epoch
- `μ = 3.986005 × 10¹⁴ m³/s²` = Earth's gravitational constant
- `ΩE = 7.2921151467 × 10⁻⁵ rad/s` = Earth's rotation rate

---
-----------------------------------------------------------------------------
 **About the projects**
 ----------------------
# Step1: Install & Import Libraries
# Step2: Load the RINEX File
# Step3: Compute Quality Metrics for Every Satellite
# Step4: Plot 1: Coverage & SNR Dashboard
# Step5: Plot 2: Data Gap Map
# Step6: Plot 3: Quality Score Summary Scatter
# Step7: Generate the Text Quality Report
=============================================================================
"""

pip install --upgrade georinex

# ───────────────────────────────────
# Step 1 — Install & Import Libraries
# ───────────────────────────────────
# Uncomment if running for the first time:
# !pip install --upgrade georinex numpy matplotlib

import georinex as gr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mpl_toolkits.mplot3d import Axes3D
import warnings
import os
os.mkdir('../content/output')

warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.dpi': 120})

# ─────────────────────────────────────────────
# GPS Physical Constants  (ICD-GPS-200)
# ─────────────────────────────────────────────
MU    = 3.986005e14          # Earth gravitational constant [m³/s²]
OMEGA_E = 7.2921151467e-5   # Earth rotation rate [rad/s]
C     = 299_792_458.0       # speed of light [m/s]
F     = -4.442807633e-10    # relativistic correction constant [s/√m]

print('✅ Libraries loaded')
print(f'   μ  (gravitational constant) : {MU:.6e} m³/s²')
print(f'   ΩE (Earth rotation rate)    : {OMEGA_E:.10e} rad/s')
print(f'   GPS orbital period          : {2*np.pi/np.sqrt(MU/(26560e3)**3)/3600:.4f} hours  (~12 h)')

# ─────────────────────────────────────────────────────────────────
# Step 2 — Load the Navigation File
# ─────────────────────────────────────────────────────────────────
# The path of the rinex navigation file
nav_path = "/brdc0010.26n"

# Load the navigation file
# georinex returns an xarray Dataset with all ephemeris parameters
print("⏳ Loading navigation file...")
nav = gr.load(nav_path)
print("✅ Navigation file loaded!")
print()
print(nav)
print()

# List all satellites (GPS = G01 to G32)
all_sv = nav.sv.values
gps_sats = sorted([s for s in all_sv if s.startswith('G')])

print(f"📡 GPS satellites in file  : {len(gps_sats)}")
print(f"   {' '.join(gps_sats)}")
print()
print(f"📋 Ephemeris parameters available:")
for var in nav.data_vars:
    print(f"   {var}")

# ───────────────────────────────────────────
# Step 3 — The GPS Orbit Computation Function
# ───────────────────────────────────────────
def gps_satellite_position(eph, t_gps):
    """
    Compute GPS satellite ECEF position from broadcast ephemeris.
    Implements the ICD-GPS-200 algorithm (9 steps).

    Parameters
    ----------
    eph   : dict-like — one ephemeris record (row from nav dataset)
    t_gps : float     — GPS time of signal transmission [seconds of GPS week]

    Returns
    -------
    X, Y, Z : float — ECEF coordinates [metres]
    """

    # ── Extract ephemeris parameters ────────────────────────────
    # Semi-major axis: sqrtA is √a, so a = sqrtA²
    sqrtA  = float(eph['sqrtA'])       # √(semi-major axis) [√m]
    a      = sqrtA ** 2                # semi-major axis [m]  (~26,560 km for GPS)

    e      = float(eph['Eccentricity'])  # eccentricity  (~0.01, nearly circular)
    M0     = float(eph['M0'])            # mean anomaly at reference epoch [rad]
    delta_n = float(eph['DeltaN'])       # mean motion correction [rad/s]
    omega  = float(eph['omega'])         # argument of perigee [rad]
    i0     = float(eph['Io'])            # inclination at reference epoch [rad]
    IDOT   = float(eph['IDOT'])          # rate of inclination [rad/s]
    Omega0 = float(eph['Omega0'])        # longitude of ascending node [rad]
    OmegaDot = float(eph['OmegaDot'])   # rate of right ascension [rad/s]
    toe    = float(eph['Toe'])           # reference time of ephemeris [s of GPS week]

    # Harmonic correction terms
    Cuc   = float(eph['Cuc'])   # argument of latitude correction (cos) [rad]
    Cus   = float(eph['Cus'])   # argument of latitude correction (sin) [rad]
    Crc   = float(eph['Crc'])   # radius correction (cos) [m]
    Crs   = float(eph['Crs'])   # radius correction (sin) [m]
    Cic   = float(eph['Cic'])   # inclination correction (cos) [rad]
    Cis   = float(eph['Cis'])   # inclination correction (sin) [rad]

    # ── Step 1 — Time from ephemeris reference epoch ─────────────
    # tk is the time elapsed since the ephemeris was valid
    tk = t_gps - toe

    # Handle GPS week crossover (toe near end/start of week)
    if tk >  302400: tk -= 604800
    if tk < -302400: tk += 604800

    # ── Step 2 — Mean anomaly ────────────────────────────────────
    # n0 = computed mean motion [rad/s]
    # Δn = correction to mean motion from broadcast (accounts for J2 perturbation)
    n0 = np.sqrt(MU / a**3)     # Keplerian mean motion [rad/s]
    n  = n0 + delta_n           # corrected mean motion [rad/s]
    Mk = M0 + n * tk            # mean anomaly at time t [rad]

    # ── Step 3 — Eccentric anomaly (Kepler's equation) ───────────
    # Kepler's equation: Ek = Mk + e·sin(Ek) — no closed-form solution
    # We solve iteratively: start with Ek = Mk, update until convergence
    Ek = Mk
    for _ in range(50):                    # 50 iterations — always enough
        Ek_new = Mk + e * np.sin(Ek)      # Newton-style iteration
        if abs(Ek_new - Ek) < 1e-12:      # convergence threshold [rad]
            break
        Ek = Ek_new
    Ek = Ek_new

    # ── Step 4 — True anomaly ────────────────────────────────────
    # True anomaly = actual angular position in the orbit
    sin_vk = (np.sqrt(1 - e**2) * np.sin(Ek)) / (1 - e * np.cos(Ek))
    cos_vk = (np.cos(Ek) - e)               / (1 - e * np.cos(Ek))
    vk     = np.arctan2(sin_vk, cos_vk)     # true anomaly [rad]

    # ── Step 5 — Argument of latitude ────────────────────────────
    Phi_k = vk + omega                      # argument of latitude [rad]

    # ── Step 6 — Second-order harmonic corrections ────────────────
    # These correct for the non-spherical shape of Earth's gravity field
    delta_uk = Cus * np.sin(2*Phi_k) + Cuc * np.cos(2*Phi_k)  # latitude correction
    delta_rk = Crs * np.sin(2*Phi_k) + Crc * np.cos(2*Phi_k)  # radius correction
    delta_ik = Cis * np.sin(2*Phi_k) + Cic * np.cos(2*Phi_k)  # inclination correction

    uk = Phi_k + delta_uk                                        # corrected argument of latitude
    rk = a * (1 - e * np.cos(Ek)) + delta_rk                   # corrected radius [m]
    ik = i0 + IDOT * tk + delta_ik                              # corrected inclination [rad]

    # ── Step 7 — Position in orbital plane ───────────────────────
    xk_prime = rk * np.cos(uk)             # x in orbital plane [m]
    yk_prime = rk * np.sin(uk)             # y in orbital plane [m]

    # ── Step 8 — Corrected longitude of ascending node ───────────
    # Earth rotates under the satellite — we account for this
    Omega_k = Omega0 + (OmegaDot - OMEGA_E) * tk - OMEGA_E * toe

    # ── Step 9 — ECEF coordinates ────────────────────────────────
    X = xk_prime * np.cos(Omega_k) - yk_prime * np.cos(ik) * np.sin(Omega_k)
    Y = xk_prime * np.sin(Omega_k) + yk_prime * np.cos(ik) * np.cos(Omega_k)
    Z = yk_prime * np.sin(ik)

    return X, Y, Z


print('✅ GPS orbit computation function defined')
print()
print('The function implements the ICD-GPS-200 algorithm in 9 steps:')
print('  1. Time from ephemeris reference epoch (tk)')
print('  2. Mean anomaly (Mk)')
print('  3. Eccentric anomaly (Ek) — iterative solution of Kepler equation')
print('  4. True anomaly (νk)')
print('  5. Argument of latitude (Φk)')
print('  6. Second-order harmonic corrections (uk, rk, ik)')
print('  7. Position in orbital plane (xk\', yk\')')
print('  8. Corrected ascending node longitude (Ωk)')
print('  9. ECEF coordinates (X, Y, Z)')

# ─────────────────────────────────────────────────────────
# Step 4 — Helper: Find the Best Ephemeris for a Given Time
# ─────────────────────────────────────────────────────────
def find_best_ephemeris(nav, sat, t_gps):
    """
    Find the ephemeris record for a satellite whose Toe is closest to t_gps.

    GPS satellites broadcast a new ephemeris set every ~2 hours.
    Each set is valid for ±2 hours around its reference epoch (Toe).
    We pick the record with the smallest |t - Toe| to minimise extrapolation error.

    Parameters
    ----------
    nav   : xarray.Dataset — full navigation dataset
    sat   : str            — satellite PRN e.g. 'G05'
    t_gps : float          — GPS seconds of week

    Returns
    -------
    eph : dict-like — the best ephemeris record, or None if not found
    """
    try:
        sat_data = nav.sel(sv=sat)
    except Exception:
        return None

    # Toe values for this satellite across all available ephemeris records
    toes = sat_data['Toe'].values

    # Find the epoch index with the smallest time difference
    best_idx = np.nanargmin(np.abs(toes - t_gps))

    # Check that the time difference is within ±7200 seconds (2 hours)
    if np.abs(toes[best_idx] - t_gps) > 7200:
        return None

    return sat_data.isel(time=best_idx)


print('✅ Ephemeris selection function defined')
print()

# ─────────────────────────────────────────────
# Quick test: compute position of G01 at 00:00 UTC
# GPS week 2399 started on 2025-12-28
# 2026-01-01 00:00 UTC = 345600 seconds into GPS week 2399
# ─────────────────────────────────────────────
t_test = 345600.0   # 2026-01-01 00:00:00 UTC in GPS seconds of week

eph_test = find_best_ephemeris(nav, 'G01', t_test)
if eph_test is not None:
    X, Y, Z = gps_satellite_position(eph_test, t_test)
    radius = np.sqrt(X**2 + Y**2 + Z**2)
    print(f"Test: G01 at 00:00 UTC")
    print(f"   X = {X/1e6:.3f} Mm")
    print(f"   Y = {Y/1e6:.3f} Mm")
    print(f"   Z = {Z/1e6:.3f} Mm")
    print(f"   Orbital radius = {radius/1e3:.1f} km  (expected ~26,560 km for GPS)")
else:
    print("G01 not found at this epoch — check your file")

# ───────────────────────────────────────────────────────────────
# Step 5 — Compute Positions for All GPS Satellites Over 24 Hours
# ───────────────────────────────────────────────────────────────
# Time grid: 00:00 to 23:45 UTC, every 15 minutes
# GPS week 2399 started 2025-12-28 00:00:00 UTC
# 2026-01-01 00:00 UTC = 4 days × 86400 s = 345600 s into the week

GPS_WEEK_START = 345600.0    # GPS seconds of week at 2026-01-01 00:00 UTC
STEP_SEC       = 900.0       # 15 minutes = 900 seconds
N_EPOCHS       = 96          # 24 hours / 15 min = 96 epochs

t_gps_epochs   = GPS_WEEK_START + np.arange(N_EPOCHS) * STEP_SEC
utc_timestamps = pd.date_range('2026-01-01 00:00', periods=N_EPOCHS, freq='15min')

print(f'Time grid: {N_EPOCHS} epochs from {utc_timestamps[0]} to {utc_timestamps[-1]}')
print(f'Step: {STEP_SEC/60:.0f} minutes')
print()

# Storage: dictionary sat → arrays of (X, Y, Z)

positions = {}   # sat → {'X': array, 'Y': array, 'Z': array, 'R': array}

print('⏳ Computing satellite positions...')
print()

for sat in gps_sats:
    X_arr = np.full(N_EPOCHS, np.nan)
    Y_arr = np.full(N_EPOCHS, np.nan)
    Z_arr = np.full(N_EPOCHS, np.nan)

    for i, t in enumerate(t_gps_epochs):
        eph = find_best_ephemeris(nav, sat, t)
        if eph is None:
            continue   # satellite not available at this epoch
        try:
            X, Y, Z = gps_satellite_position(eph, t)
            X_arr[i] = X
            Y_arr[i] = Y
            Z_arr[i] = Z
        except Exception:
            pass

    # Orbital radius (distance from Earth's centre)
    R_arr = np.sqrt(X_arr**2 + Y_arr**2 + Z_arr**2)

    n_valid = np.sum(~np.isnan(X_arr))
    if n_valid > 0:
        r_mean = np.nanmean(R_arr) / 1e3
        positions[sat] = {'X': X_arr, 'Y': Y_arr, 'Z': Z_arr, 'R': R_arr}
        print(f'  {sat}: {n_valid:>3} valid epochs  |  mean radius = {r_mean:.1f} km')

print()
print(f'✅ Done!  {len(positions)} satellites computed')

# ────────────────────────────────────────────
# Step 6 — Plot 1: 3D GPS Constellation Orbits
# ────────────────────────────────────────────
# Earth radius [m] for the wireframe sphere
R_EARTH = 6_371_000.0

# Colour palette for the satellites
palette = plt.cm.plasma(np.linspace(0.1, 0.9, len(positions)))
sat_list = list(positions.keys())

# Build Earth wireframe sphere

theta = np.linspace(0, 2*np.pi, 60)
phi   = np.linspace(0, np.pi,   30)
xe = R_EARTH * np.outer(np.cos(theta), np.sin(phi)) / 1e6
ye = R_EARTH * np.outer(np.sin(theta), np.sin(phi)) / 1e6
ze = R_EARTH * np.outer(np.ones_like(theta), np.cos(phi)) / 1e6

# Figure

fig = plt.figure(figsize=(14, 12), facecolor='#0d1117')
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#0d1117')

# Earth wireframe
ax.plot_wireframe(
    xe, ye, ze,
    color='#1565C0', alpha=0.25, linewidth=0.4, rstride=2, cstride=2
)

# Equatorial plane reference circle
eq_angle = np.linspace(0, 2*np.pi, 200)
r_orbit  = np.nanmean([np.nanmean(positions[s]['R']) for s in sat_list]) / 1e6
ax.plot(r_orbit*np.cos(eq_angle), r_orbit*np.sin(eq_angle), np.zeros(200),
        color='#37474F', lw=0.8, ls='--', alpha=0.5)

# Plot each satellite orbit
for i, sat in enumerate(sat_list):
    X = positions[sat]['X'] / 1e6   # convert m → Mm for readability
    Y = positions[sat]['Y'] / 1e6
    Z = positions[sat]['Z'] / 1e6

    # Plot the orbit arc (skip NaN gaps)
    mask = ~np.isnan(X)
    ax.plot(X[mask], Y[mask], Z[mask],
            color=palette[i], lw=1.2, alpha=0.85)

    # Mark the satellite's position at 00:00 UTC
    if not np.isnan(X[0]):
        ax.scatter(X[0], Y[0], Z[0],
                   color=palette[i], s=20, zorder=10)
        ax.text(X[0], Y[0], Z[0], f' {sat[1:]}',   # label without 'G' prefix
                fontsize=5, color=palette[i], alpha=0.9)

# Axis labels and styling
ax.set_xlabel('X [Mm]', color='#aaaaaa', fontsize=9, labelpad=8)
ax.set_ylabel('Y [Mm]', color='#aaaaaa', fontsize=9, labelpad=8)
ax.set_zlabel('Z [Mm]', color='#aaaaaa', fontsize=9, labelpad=8)

ax.tick_params(colors='#555555', labelsize=7)
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('#1a1a2e')
ax.yaxis.pane.set_edgecolor('#1a1a2e')
ax.zaxis.pane.set_edgecolor('#1a1a2e')
ax.grid(True, color='#1a1a2e', linewidth=0.4)

ax.set_title(
    f'GPS Constellation Orbits — {len(positions)} Satellites\n'
    'Computed from Broadcast Ephemeris | 2026-01-01 | ECEF Frame\n'
    'Dots = position at 00:00 UTC  |  Dashed circle = equatorial reference',
    fontsize=12, fontweight='bold', color='#ffffff', pad=15
)

plt.tight_layout()
plt.savefig('../content/output/plot1_3d_orbits.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()

print('✅ Saved: output/plot1_3d_orbits.png')
print()
print('💡 Interpretation:')
print('   • All GPS satellites orbit in 6 orbital planes (55° inclination)')
print('   • Each orbit takes ~11h 58m (half a sidereal day)')
print(f'   • Mean orbital radius: {r_orbit:.1f} Mm = {r_orbit*1000:.0f} km  (expected ~26,560 km)')
print('   • The ECEF frame rotates with the Earth — so orbit curves are NOT the true inertial orbits')

# ────────────────────────────────
# Step 7 — Plot 2: XYZ Time Series
# ────────────────────────────────
# Select a few satellites to keep the plot readable
SELECTED = [s for s in ['G01', 'G05', 'G10', 'G15', 'G20', 'G25'] if s in positions]

fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True, facecolor='#0d1117')
fig.suptitle(
    'GPS Satellite ECEF Coordinates Over 24 Hours\n'
    f'Selected satellites: {", ".join(SELECTED)} | 2026-01-01',
    fontsize=13, fontweight='bold', color='#ffffff'
)

ylabels = ['X [Mm]', 'Y [Mm]', 'Z [Mm]']
keys    = ['X', 'Y', 'Z']

for ax, key, ylabel in zip(axes, keys, ylabels):
    ax.set_facecolor('#111827')
    ax.tick_params(colors='#aaaaaa')
    ax.grid(True, color='#222222', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')
    ax.axhline(0, color='#444444', lw=0.8, ls='--')
    ax.set_ylabel(ylabel, color='#aaaaaa', fontsize=11)

    colors = plt.cm.tab10(np.linspace(0, 1, len(SELECTED)))
    for sat, color in zip(SELECTED, colors):
        vals = positions[sat][key] / 1e6   # m → Mm
        mask = ~np.isnan(vals)
        ax.plot(utc_timestamps[mask], vals[mask],
                color=color, lw=1.6, alpha=0.9, label=sat)

    ax.set_title(
        f'ECEF {key} coordinate — oscillates because the ECEF frame rotates with Earth',
        color='white', fontsize=10
    )

    legend = ax.legend(
        ncol=len(SELECTED), fontsize=9, loc='upper right',
        framealpha=0.3, facecolor='#1a1a2e', edgecolor='#444444'
    )
    for t in legend.get_texts():
        t.set_color('white')

axes[-1].set_xlabel('UTC Time (HH:MM)', color='#aaaaaa', fontsize=11)
axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axes[-1].xaxis.set_major_locator(mdates.HourLocator(interval=2))
plt.xticks(rotation=30, color='#aaaaaa')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('../content/output/plot2_xyz_timeseries.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()

print('✅ Saved: output/plot2_xyz_timeseries.png')
print()
print('💡 Interpretation:')
print('   • X and Y oscillate with ~12h period (orbital period in the rotating ECEF frame)')
print('   • Z oscillates because the orbit is inclined at ~55° to the equator')
print('   • Z = 0 when the satellite is crossing the equatorial plane')
print('   • The amplitude of XYZ variations = orbital radius (~26,560 km)')

# ────────────────────────────────────────────
# Step 8 — Plot 3: Orbital Radius Verification
# ────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(16, 9), facecolor='#0d1117')
fig.suptitle(
    'GPS Orbital Radius Verification\n'
    'R = √(X²+Y²+Z²) — Expected ~26,560 km for all GPS satellites',
    fontsize=13, fontweight='bold', color='#ffffff'
)

for ax in axes:
    ax.set_facecolor('#111827')
    ax.tick_params(colors='#aaaaaa')
    ax.grid(True, color='#222222', linewidth=0.5)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')

palette2 = plt.cm.tab20(np.linspace(0, 1, len(positions)))

# ─── Panel 1: Radius time series for all satellites ──────────
for i, sat in enumerate(sat_list):
    R = positions[sat]['R'] / 1e3   # m → km
    mask = ~np.isnan(R)
    axes[0].plot(utc_timestamps[mask], R[mask],
                 color=palette2[i], lw=1.2, alpha=0.75, label=sat)

# Reference line at nominal GPS orbital radius
axes[0].axhline(26560, color='#FFEB3B', ls='--', lw=1.5,
                label='Nominal GPS radius (26,560 km)')

axes[0].set_ylabel('Orbital Radius [km]', color='#aaaaaa', fontsize=11)
axes[0].set_title(
    'Orbital radius per satellite over 24 hours (all should cluster near 26,560 km)',
    color='white', fontsize=10
)
axes[0].set_ylim(26000, 27000)

legend0 = axes[0].legend(
    ncol=4, fontsize=7, loc='lower right',
    framealpha=0.3, facecolor='#1a1a2e', edgecolor='#444444'
)
for t in legend0.get_texts(): t.set_color('white')

axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axes[0].xaxis.set_major_locator(mdates.HourLocator(interval=2))

# ─── Panel 2: Mean radius per satellite (bar chart) ──────────
mean_radii = {sat: np.nanmean(positions[sat]['R'])/1e3 for sat in sat_list}
std_radii  = {sat: np.nanstd( positions[sat]['R'])/1e3 for sat in sat_list}

sats_s = list(mean_radii.keys())
means  = list(mean_radii.values())
stds   = list(std_radii.values())
x_pos  = np.arange(len(sats_s))

bars = axes[1].bar(
    x_pos, means,
    color=[palette2[i] for i in range(len(sats_s))],
    edgecolor='#0d1117', linewidth=0.8,
    yerr=stds, capsize=3,
    error_kw=dict(ecolor='white', elinewidth=0.8, alpha=0.6)
)

axes[1].axhline(26560, color='#FFEB3B', ls='--', lw=1.5,
                label='Nominal GPS radius (26,560 km)')
axes[1].set_xticks(x_pos)
axes[1].set_xticklabels(sats_s, rotation=45, ha='right',
                         color='#aaaaaa', fontsize=8)
axes[1].set_ylabel('Mean Orbital Radius [km]', color='#aaaaaa', fontsize=11)
axes[1].set_title(
    'Mean orbital radius per satellite (error bars = ±1σ variation due to eccentricity)',
    color='white', fontsize=10
)
axes[1].set_ylim(26400, 26700)

legend1 = axes[1].legend(fontsize=9, framealpha=0.3,
                          facecolor='#1a1a2e', edgecolor='#444444')
for t in legend1.get_texts(): t.set_color('white')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('../content/output/plot3_orbital_radius.png',
            dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.show()

print('✅ Saved: output/plot3_orbital_radius.png')
print()
print('📊 Orbital radius summary:')
print(f"   {'Satellite':<8} {'Mean [km]':>10} {'Std [km]':>10} {'Min [km]':>10} {'Max [km]':>10}")
print("   " + "-" * 50)
for sat in sat_list:
    R = positions[sat]['R'] / 1e3
    print(f"   {sat:<8} {np.nanmean(R):>10.2f} {np.nanstd(R):>10.2f} "
          f"{np.nanmin(R):>10.2f} {np.nanmax(R):>10.2f}")
print()
print('💡 The small variation (~10–30 km) is due to orbital eccentricity (e ≈ 0.01)')
print('   GPS orbits are not perfectly circular — they are slightly elliptical.')

