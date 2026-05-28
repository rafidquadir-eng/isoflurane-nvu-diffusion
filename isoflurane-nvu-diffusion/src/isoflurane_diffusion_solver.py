"""
Isoflurane Diffusion in the Neurovascular Unit
===============================================
Multi-compartment cylindrical PDE solver (implicit finite differences)

Model geometry — concentric cylindrical shells:
    [Vessel lumen] -> [BBB wall] -> [ECS/parenchyma] -> [Neuron]
      r = 0           R_v = 5 um    R_w = 6 um          R_n = 15 um

Governing equation per compartment i:
    dc/dt = (D_eff_i / r) * d/dr(r * dc/dr) - k_i * c

Boundary conditions:
    - Dirichlet at vessel wall: c(R_v) = c_blood * K_bt
    - Permeability-limited flux at neuron membrane
    - Zero-flux (Neumann) at outer domain boundary

References:
    - Balluffi, Allen, Carter. Kinetics of Materials. Wiley, 2005.
    - Franks NP (2008) Molecular targets underlying general anaesthesia.
      Br J Pharmacol 147:S72-S81.

Author : [Your name]
Lab    : [Lab name] — Neuroscience, [Institution]
Date   : Originally conceived ~2019; implemented May 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erf
from scipy.linalg import solve_banded


# ── 1. PARAMETERS ─────────────────────────────────────────────────────────────

# --- Geometry (SI units: meters) ---
R_v   = 5e-6    # vessel inner wall radius
R_w   = 6e-6    # BBB outer radius (~1 um wall)
R_n   = 15e-6   # neuron outer radius
R_max = 30e-6   # domain outer boundary

# --- Diffusion coefficients (m²/s) ---
D_blood = 5e-10   # plasma (protein-binding corrected, f_free ~ 0.6)
D_BBB   = 1e-11   # blood-brain barrier (lipid bilayer, lipophilicity-enhanced)
D_ECS   = 3e-10   # extracellular space (tortuosity lambda=1.6, D_free/lambda²)
D_cyto  = 1e-10   # neuronal cytoplasm (crowding + organelles)

# --- First-order binding/sink rates (1/s) ---
k_blood = 0.0     # plasma: protein binding handled in D correction
k_BBB   = 1e-2    # BBB lipid nonspecific binding
k_ECS   = 1e-3    # ECS: mild nonspecific binding
k_cyto  = 5e-3    # cytoplasm: membrane and organelle binding

# --- Partition coefficients ---
K_bt = 1.4    # brain tissue : blood
K_mt = 1.2    # neuron membrane : ECS

# --- Isoflurane boundary condition (1 MAC) ---
c_MAC      = 0.23e-3          # mol/m³ at 1 MAC in blood
c_vessel   = c_MAC * K_bt    # concentration at vessel wall inner surface

# --- Membrane permeability (m/s) ---
P_m = 1e-5   # high — isoflurane is lipophilic

# --- Time parameters ---
t_end   = 300.0              # 5 minutes (s)
dt      = 0.05               # time step (s)
n_steps = int(t_end / dt)
t_save  = [10, 30, 60, 120, 300]  # snapshot times (s)

# --- Spatial grid ---
N  = 800
r  = np.linspace(1e-9, R_max, N)   # avoid r=0 singularity
dr = r[1] - r[0]


# ── 2. COMPARTMENT MAPS ───────────────────────────────────────────────────────

def D_of_r(r_val):
    """Return diffusion coefficient at radius r_val."""
    if   r_val < R_v: return D_blood
    elif r_val < R_w: return D_BBB
    elif r_val < R_n: return D_ECS
    else:             return D_cyto

def k_of_r(r_val):
    """Return sink rate at radius r_val."""
    if   r_val < R_v: return k_blood
    elif r_val < R_w: return k_BBB
    elif r_val < R_n: return k_ECS
    else:             return k_cyto

D_arr = np.array([D_of_r(ri) for ri in r])
k_arr = np.array([k_of_r(ri) for ri in r])

# Compartment boundary indices
idx_v = np.searchsorted(r, R_v)
idx_w = np.searchsorted(r, R_w)
idx_n = np.searchsorted(r, R_n)


# ── 3. IMPLICIT FINITE DIFFERENCE MATRIX ──────────────────────────────────────
# Cylindrical Laplacian discretization:
#   dc/dt = D_i/r_i * [r_{i+1/2}(c_{i+1}-c_i) - r_{i-1/2}(c_i-c_{i-1})] / dr²
#         - k_i * c_i
#
# Implicit Euler => tridiagonal system solved each step.
# Harmonic mean D at interfaces handles discontinuities between compartments.

def build_banded_matrix(r, D_arr, k_arr, dr, dt, N):
    """
    Build banded (3 x N) matrix for scipy.linalg.solve_banded (1,1).
    Row 0 = upper diagonal, Row 1 = main diagonal, Row 2 = lower diagonal.
    """
    r_ph = 0.5 * (r[:-1] + r[1:])                              # r_{i+1/2}, length N-1
    r_mh = np.concatenate(([max(r[0] - dr/2, 1e-9)], r_ph))  # r_{i-1/2}, length N

    # Harmonic mean D at half-grid interfaces (handles jump discontinuities)
    D_ph = 2*D_arr[:-1]*D_arr[1:] / (D_arr[:-1] + D_arr[1:] + 1e-40)  # length N-1
    D_mh = np.concatenate(([D_arr[0]], D_ph))                           # length N

    alpha = dt * D_mh * r_mh / (r * dr**2)        # sub-diagonal coeff, length N
    beta  = np.concatenate((dt * D_ph * r_ph / (r[:-1] * dr**2), [0.0]))  # length N

    ab = np.zeros((3, N))
    ab[0, 1:]  = -beta[:-1]          # upper diagonal
    ab[1, :]   = 1 + alpha + beta + dt * k_arr   # main diagonal
    ab[2, :-1] = -alpha[1:]          # lower diagonal

    return ab

ab = build_banded_matrix(r, D_arr, k_arr, dr, dt, N)


# ── 4. BOUNDARY CONDITION APPLICATION ─────────────────────────────────────────

def apply_BC(c, ab_base):
    """
    Returns modified (ab, rhs) with BCs applied.

    BCs:
      - Dirichlet: vessel lumen held at c_vessel (source)
      - Robin (permeability) at neuron surface: -D_ECS * dc/dr = P_m*(c - c_in/K_mt)
      - Neumann (zero flux) at outer domain boundary
    """
    ab  = ab_base.copy()
    rhs = c.copy()

    # --- Dirichlet: vessel lumen ---
    for i in range(idx_v + 1):
        ab[0, i+1] = 0.0 if i+1 < N else 0.0
        ab[1, i]   = 1.0
        if i > 0:
            ab[2, i-1] = 0.0
        rhs[i] = c_vessel

    # --- Neumann: zero flux at outer boundary ---
    ab[1, -1]  =  1.0
    ab[2, -2]  = -1.0
    rhs[-1]    =  0.0

    return ab, rhs


# ── 5. TIME INTEGRATION ───────────────────────────────────────────────────────

c = np.zeros(N)
# Initialise vessel lumen at source concentration
c[:idx_v+1] = c_vessel

snapshots = {}
t = 0.0

print("Running simulation...")
for step in range(n_steps):
    t += dt
    ab_mod, rhs = apply_BC(c, ab)
    c = solve_banded((1, 1), ab_mod, rhs)

    for ts in t_save:
        if abs(t - ts) < dt / 2:
            snapshots[ts] = c.copy()

print("Done.\n")


# ── 6. ANALYTICAL ERF SOLUTION (ECS, early-time approximation) ────────────────

def c_erf_approx(r_vals, t_val, c0=c_vessel, D=D_ECS, r0=R_w):
    """
    Semi-infinite planar erf approximation valid for early times
    when diffusion front has not reached neuron surface.
    From: c(r,t) = c0 * [1 - erf((r-r0)/sqrt(4*D*t))]
    """
    xi = (r_vals - r0) / np.sqrt(4 * D * t_val + 1e-40)
    return c0 * np.maximum(0, 1 - erf(xi))

# Diffusion distance estimate: x ~ sqrt(4*D_ECS*t)
print("Diffusion distance estimates (ECS):")
for ts in t_save:
    x_diff = np.sqrt(4 * D_ECS * ts) * 1e6
    print(f"  t = {ts:4.0f} s  =>  x ~ {x_diff:.2f} um")

print("\nConcentration at neuron surface (r = R_n):")
for ts in t_save:
    if ts in snapshots:
        c_at_n = snapshots[ts][idx_n]
        pct    = c_at_n / c_vessel * 100
        print(f"  t = {ts:4.0f} s  =>  {pct:.1f}% of vessel concentration")


# ── 7. FIGURES ────────────────────────────────────────────────────────────────

colors    = plt.cm.plasma(np.linspace(0.15, 0.92, len(t_save)))
r_um      = r * 1e6
r_ecs_mask = (r >= R_w) & (r <= R_n)
r_ecs_um   = r[r_ecs_mask] * 1e6

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# ── Panel A: Full radial concentration profiles ──
ax1 = fig.add_subplot(gs[0, :2])
for i, ts in enumerate(t_save):
    if ts in snapshots:
        ax1.plot(r_um, snapshots[ts] / c_vessel,
                 color=colors[i], lw=2, label=f't = {ts} s')

ax1.axvspan(0,       R_v*1e6, alpha=0.08, color='red',   label='Vessel lumen')
ax1.axvspan(R_v*1e6, R_w*1e6, alpha=0.15, color='orange',label='BBB')
ax1.axvspan(R_w*1e6, R_n*1e6, alpha=0.08, color='green', label='ECS')
ax1.axvspan(R_n*1e6, R_max*1e6, alpha=0.08, color='blue',label='Neuron')

ax1.set_xlabel('Radial distance r (μm)', fontsize=12)
ax1.set_ylabel('c / c_vessel', fontsize=12)
ax1.set_title('A  |  Multi-compartment radial concentration profiles', fontsize=12, fontweight='bold')
ax1.legend(fontsize=8, ncol=2, loc='upper right')
ax1.set_xlim([0, R_max*1e6])
ax1.set_ylim([0, 1.15])
ax1.grid(alpha=0.25)

# ── Panel B: ECS zoom — numerical vs erf analytical ──
ax2 = fig.add_subplot(gs[0, 2])
for i, ts in enumerate(t_save):
    if ts in snapshots:
        c_num = snapshots[ts][r_ecs_mask]
        c_an  = c_erf_approx(r[r_ecs_mask], ts)
        ax2.plot(r_ecs_um, c_num / c_vessel,
                 color=colors[i], lw=2)
        ax2.plot(r_ecs_um, c_an / c_vessel,
                 color=colors[i], lw=1.2, ls=':', alpha=0.7)

ax2.set_xlabel('r (μm)', fontsize=11)
ax2.set_ylabel('c / c_vessel', fontsize=11)
ax2.set_title('B  |  ECS zoom\nNumerical (—) vs erf (···)', fontsize=11, fontweight='bold')
ax2.grid(alpha=0.25)

# ── Panel C: Concentration at neuron surface vs time ──
ax3 = fig.add_subplot(gs[1, 0])
t_fine    = np.linspace(dt, t_end, n_steps)
c_neuron  = np.array([snapshots.get(ts, np.zeros(N))[idx_n]
                       for ts in t_save]) / c_vessel
ts_arr    = np.array([ts for ts in t_save if ts in snapshots])
ax3.plot(ts_arr, c_neuron[:len(ts_arr)]*100, 'o-', color='purple', lw=2, ms=6)
ax3.axhline(50, color='gray', ls='--', alpha=0.6, label='50% threshold')
ax3.set_xlabel('Time (s)', fontsize=11)
ax3.set_ylabel('c at neuron / c_vessel  (%)', fontsize=10)
ax3.set_title('C  |  Neuron surface uptake', fontsize=11, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.25)

# ── Panel D: Diffusion distance vs time ──
ax4 = fig.add_subplot(gs[1, 1])
t_arr  = np.linspace(1, t_end, 300)
x_diff = np.sqrt(4 * D_ECS * t_arr) * 1e6
ax4.plot(t_arr, x_diff, color='teal', lw=2)
ax4.axhline((R_n - R_w)*1e6, color='green', ls='--', label=f'ECS width ({(R_n-R_w)*1e6:.0f} μm)')
ax4.fill_between(t_arr, x_diff, alpha=0.15, color='teal')
ax4.set_xlabel('Time (s)', fontsize=11)
ax4.set_ylabel('Diffusion distance (μm)', fontsize=11)
ax4.set_title('D  |  x ≈ √(4 D_ECS t)', fontsize=11, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.25)

# ── Panel E: D(r) and k(r) compartment map ──
ax5 = fig.add_subplot(gs[1, 2])
ax5b = ax5.twinx()
ax5.semilogy(r_um, D_arr, color='navy', lw=2, label='D_eff(r)')
ax5b.semilogy(r_um, k_arr + 1e-10, color='crimson', lw=1.5, ls='--', label='k(r)')
ax5.axvspan(0,       R_v*1e6,   alpha=0.08, color='red')
ax5.axvspan(R_v*1e6, R_w*1e6,   alpha=0.15, color='orange')
ax5.axvspan(R_w*1e6, R_n*1e6,   alpha=0.08, color='green')
ax5.axvspan(R_n*1e6, R_max*1e6, alpha=0.08, color='blue')
ax5.set_xlabel('r (μm)', fontsize=11)
ax5.set_ylabel('D_eff (m²/s)', fontsize=10, color='navy')
ax5b.set_ylabel('k (1/s)', fontsize=10, color='crimson')
ax5.set_title('E  |  Compartment properties', fontsize=11, fontweight='bold')
lines1, labs1 = ax5.get_legend_handles_labels()
lines2, labs2 = ax5b.get_legend_handles_labels()
ax5.legend(lines1+lines2, labs1+labs2, fontsize=8)
ax5.grid(alpha=0.25)

fig.suptitle(
    'Isoflurane Diffusion — Neurovascular Unit\n'
    'Multi-compartment cylindrical PDE  |  Implicit finite differences  |  v0.1',
    fontsize=13, fontweight='bold', y=1.01
)

out_path = '/mnt/user-data/outputs/isoflurane_diffusion.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure saved: {out_path}")
