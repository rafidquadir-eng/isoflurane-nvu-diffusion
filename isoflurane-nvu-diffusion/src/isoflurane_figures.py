"""
Isoflurane Diffusion — 2D Visualization + Sensitivity Analysis
==============================================================
Generates two publication-quality figures:
  1. isoflurane_2d_field.png   — 2D multi-vessel concentration field (dark theme)
  2. isoflurane_sensitivity.png — sensitivity analysis: 4 parameters × 5 values

Author: [Your name]
Date  : May 2026
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Patch
from scipy.linalg import solve_banded
from scipy.interpolate import interp1d

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9})

# ── SHARED PARAMETERS ─────────────────────────────────────────────────────────

R_w_offset = 1e-6    # BBB thickness
R_n_base   = 15e-6   # neuron outer radius
R_max      = 38e-6   # simulation domain

D_blood   = 5e-10
D_BBB_0   = 1e-11
D_BBB_max = 1e-10
D_cyto    = 1e-10
k_blood, k_BBB, k_ECS, k_cyto = 0.0, 1e-2, 1e-3, 5e-3
c_MAC = 0.23e-3

# Baseline
R_v_base    = 5e-6
D_ECS_base  = 3e-10
K_bt_base   = 1.4
n_hill_base = 2.0


# ── SIMULATION ENGINE ─────────────────────────────────────────────────────────

def run_sim(R_v_p=R_v_base, D_ECS_p=D_ECS_base, K_bt_p=K_bt_base,
            n_hill_p=n_hill_base, t_end=300.0, t_saves=None, N=400, dt=0.1):
    if t_saves is None:
        t_saves = [10, 60, 300]

    R_w      = R_v_p + R_w_offset
    c_vessel = c_MAC * K_bt_p
    K_half   = c_vessel / (3.5 ** (1.0 / n_hill_p))

    r  = np.linspace(1e-9, R_max, N)
    dr = r[1] - r[0]

    idx_v = np.searchsorted(r, R_v_p)
    idx_n = np.searchsorted(r, R_n_base)

    def D_BBB_c(cl):
        cl = np.maximum(cl, 0.0)
        h  = cl**n_hill_p / (K_half**n_hill_p + cl**n_hill_p + 1e-40)
        return D_BBB_0 + (D_BBB_max - D_BBB_0) * h

    k_arr = np.where(r < R_v_p, k_blood,
            np.where(r < R_w,   k_BBB,
            np.where(r < R_n_base, k_ECS, k_cyto)))

    def build_D(c):
        D = np.where(r < R_v_p, D_blood,
            np.where(r < R_w,   D_BBB_0,
            np.where(r < R_n_base, D_ECS_p, D_cyto)))
        mask  = (r >= R_v_p) & (r < R_w)
        c_avg = np.mean(c[mask]) if mask.any() else 0.0
        return np.where(mask, D_BBB_c(c_avg), D)

    def build_ab(D):
        r_ph  = 0.5 * (r[:-1] + r[1:])
        r_mh  = np.concatenate(([max(r[0]-dr/2, 1e-9)], r_ph))
        D_ph  = 2*D[:-1]*D[1:] / (D[:-1]+D[1:]+1e-40)
        D_mh  = np.concatenate(([D[0]], D_ph))
        alpha = dt * D_mh * r_mh / (r * dr**2)
        beta  = np.concatenate((dt*D_ph*r_ph/(r[:-1]*dr**2), [0.0]))
        ab    = np.zeros((3, N))
        ab[0,1:]  = -beta[:-1]
        ab[1,:]   =  1 + alpha + beta + dt*k_arr
        ab[2,:-1] = -alpha[1:]
        return ab

    def apply_BC(c, ab0):
        ab, rhs = ab0.copy(), c.copy()
        for i in range(idx_v + 1):
            if i+1 < N: ab[0,i+1] = 0.0
            ab[1,i] = 1.0
            if i > 0:   ab[2,i-1] = 0.0
            rhs[i] = c_vessel
        ab[1,-1] = 1.0; ab[2,-2] = -1.0; rhs[-1] = 0.0
        return ab, rhs

    c = np.zeros(N); c[:idx_v+1] = c_vessel
    snaps = {}
    t = 0.0
    for _ in range(int(t_end / dt)):
        t += dt
        ab_m, rhs = apply_BC(c, build_ab(build_D(c)))
        c = solve_banded((1,1), ab_m, rhs)
        for ts in t_saves:
            if abs(t - ts) < dt/2:
                snaps[ts] = c.copy()

    return r, snaps, c_vessel, idx_n


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 2D NEUROVASCULAR FIELD
# ════════════════════════════════════════════════════════════════════════════════

print("Running base simulation (for 2D visualization)...")
r_base, snaps_viz, cv_base, _ = run_sim(t_saves=[10, 60, 300])
print("  Done.")

# Vessel network: 3 line segments (inspired by the branching image)
vessel_segs = [
    (np.array([-30e-6,  1e-6]), np.array([ 30e-6, -1e-6])),   # main horizontal
    (np.array([ -7e-6,  0e-6]), np.array([-22e-6, 22e-6])),   # upper-left branch
    (np.array([  6e-6,  0e-6]), np.array([ 20e-6,-22e-6])),   # lower-right branch
]

# Neuron nuclei — placed realistically in the parenchyma
neuron_xy = np.array([
    [ 14e-6,  16e-6], [-13e-6,  20e-6], [ 22e-6,  19e-6],
    [-22e-6,   3e-6], [  9e-6, -20e-6], [-22e-6, -12e-6],
    [ 28e-6,   8e-6], [ -8e-6,  28e-6], [ 18e-6, -10e-6],
])
neuron_r = 4.5e-6


def min_dist_field(X, Y, segs):
    """Minimum distance from each grid point to the vessel network."""
    md = np.full(X.shape, np.inf)
    for A, B in segs:
        AB   = B - A
        AB2  = float(np.dot(AB, AB))
        AX, AY = X - A[0], Y - A[1]
        t  = np.clip((AX*AB[0] + AY*AB[1]) / (AB2 + 1e-40), 0, 1)
        cx = A[0] + t*AB[0]; cy = A[1] + t*AB[1]
        md = np.minimum(md, np.sqrt((X-cx)**2 + (Y-cy)**2))
    return md


def concentration_field(snap, r_sim, c_vessel_val, dist_map, R_v_p=R_v_base):
    """Map 1D radial solution to 2D via distance to vessel network."""
    c_norm = snap / c_vessel_val
    interp = interp1d(r_sim, c_norm,
                      bounds_error=False,
                      fill_value=(float(c_norm[0]), float(c_norm[-1])))
    field  = interp(dist_map)
    field  = np.where(dist_map < R_v_p, 1.0, field)   # vessel interior
    return np.clip(field, 0, 1)


# Build 2D grid
ng   = 520
x_g  = np.linspace(-36e-6, 36e-6, ng)
y_g  = np.linspace(-32e-6, 32e-6, ng)
XG, YG = np.meshgrid(x_g, y_g)
dist_map = min_dist_field(XG, YG, vessel_segs)

t_panels = [10, 60, 300]
panel_sub = ['Early phase\nBBB opening (D_BBB → 8× baseline)',
             'Mid phase\nECS equilibrating',
             'Steady state\nFull neuron uptake']

fig1, axes1 = plt.subplots(1, 3, figsize=(17, 6.2))
fig1.patch.set_facecolor('#0a0a0a')
plt.subplots_adjust(left=0.04, right=0.91, bottom=0.09, top=0.88, wspace=0.06)

# Custom colormap: black → deep navy → purple → orange → gold
from matplotlib.colors import LinearSegmentedColormap
drug_cmap = LinearSegmentedColormap.from_list(
    'isoflurane',
    [(0.00, '#050510'), (0.25, '#1a0a3d'), (0.50, '#7b2d8b'),
     (0.75, '#e06010'), (0.90, '#f5b800'), (1.00, '#fff5cc')], N=512)

ims1 = []
for ax, ts, sub in zip(axes1, t_panels, panel_sub):
    ax.set_facecolor('#0a0a0a')
    if ts in snaps_viz:
        field = concentration_field(snaps_viz[ts], r_base, cv_base, dist_map)
        im = ax.imshow(field, origin='lower',
                       extent=[x_g[0]*1e6, x_g[-1]*1e6, y_g[0]*1e6, y_g[-1]*1e6],
                       cmap=drug_cmap, vmin=0, vmax=1,
                       interpolation='bilinear', aspect='equal')
        ims1.append(im)

    # Vessel walls — draw as thick luminous outlines
    for A, B in vessel_segs:
        xs = [A[0]*1e6, B[0]*1e6]; ys = [A[1]*1e6, B[1]*1e6]
        ax.plot(xs, ys, color='white',     lw=R_v_base*2e6, alpha=0.18,
                solid_capstyle='round')                          # glow
        ax.plot(xs, ys, color='#cccccc',   lw=1.8, alpha=0.6,
                solid_capstyle='round')                          # wall outline
        ax.plot(xs, ys, color='#ff8c40',   lw=0.6, alpha=0.45,
                solid_capstyle='round', ls='--')                 # BBB highlight

    # Neuron nuclei — green blobs echoing the fluorescence image
    for nx_, ny_ in neuron_xy:
        # Outer glow
        ax.add_patch(Circle((nx_*1e6, ny_*1e6), neuron_r*1.4e6,
                             facecolor='#003320', edgecolor='none', alpha=0.5))
        # Nucleus
        ax.add_patch(Circle((nx_*1e6, ny_*1e6), neuron_r*1e6,
                             facecolor='#00cc66', edgecolor='#66ffaa',
                             lw=0.7, alpha=0.88))

    ax.set_xlim([x_g[0]*1e6, x_g[-1]*1e6])
    ax.set_ylim([y_g[0]*1e6, y_g[-1]*1e6])
    ax.set_title(f't = {ts} s\n{sub}',
                 color='white', fontsize=10.5, fontweight='bold', pad=7)
    ax.set_xlabel('x  (μm)', color='#aaaaaa', fontsize=9)
    if ax is axes1[0]:
        ax.set_ylabel('y  (μm)', color='#aaaaaa', fontsize=9)
    ax.tick_params(colors='#888888', labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_edgecolor('#333333')

# Shared colorbar
cax1 = fig1.add_axes([0.922, 0.12, 0.013, 0.72])
cb1  = fig1.colorbar(ims1[-1], cax=cax1)
cb1.set_label('Isoflurane  (fraction of 1 MAC)', color='#cccccc', fontsize=9.5, labelpad=8)
cb1.ax.yaxis.set_tick_params(color='#888888')
plt.setp(cb1.ax.yaxis.get_ticklabels(), color='#cccccc')
cb1.outline.set_edgecolor('#444444')

# Legend
legend_el = [Patch(facecolor='#888888', edgecolor='white',    label='Blood vessel / BBB'),
             Patch(facecolor='#00cc66', edgecolor='#66ffaa',  label='Neuron nucleus')]
axes1[0].legend(handles=legend_el, loc='lower left',
                facecolor='#1a1a1a', edgecolor='#555555',
                labelcolor='#cccccc', fontsize=8, framealpha=0.85)

fig1.suptitle(
    'Isoflurane Concentration Field — Multi-vessel Neurovascular Network\n'
    'Concentration-dependent  D_BBB(c)  ·  Cylindrical superposition  ·  1 MAC delivered',
    color='white', fontsize=12, fontweight='bold', y=0.98)

out1 = '/mnt/user-data/outputs/isoflurane_2d_field.png'
fig1.savefig(out1, dpi=160, bbox_inches='tight', facecolor='#0a0a0a')
plt.close(fig1)
print(f"Figure 1 saved: {out1}")


# ════════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — SENSITIVITY ANALYSIS
# ════════════════════════════════════════════════════════════════════════════════

t_saves_s = list(range(5, 125, 5))   # every 5 s up to 120 s

param_sweeps = {
    'D_ECS  (m²/s)': {
        'key': 'D_ECS_p',
        'values': [1e-10, 2e-10, 3e-10, 4e-10, 5e-10],
        'labels': ['1×10⁻¹⁰', '2×10⁻¹⁰', '3×10⁻¹⁰ ★', '4×10⁻¹⁰', '5×10⁻¹⁰'],
        'baseline_idx': 2,
        'cmap': plt.cm.Blues,
        'accent': '#2196F3',
    },
    'R_vessel  (μm)': {
        'key': 'R_v_p',
        'values': [2e-6, 3.5e-6, 5e-6, 7e-6, 10e-6],
        'labels': ['2', '3.5', '5 ★', '7', '10'],
        'baseline_idx': 2,
        'cmap': plt.cm.Oranges,
        'accent': '#FF9800',
    },
    'K_bt  (brain:blood)': {
        'key': 'K_bt_p',
        'values': [0.8, 1.1, 1.4, 1.8, 2.2],
        'labels': ['0.8', '1.1', '1.4 ★', '1.8', '2.2'],
        'baseline_idx': 2,
        'cmap': plt.cm.Greens,
        'accent': '#4CAF50',
    },
    'n_Hill  (BBB coeff)': {
        'key': 'n_hill_p',
        'values': [1.0, 1.5, 2.0, 2.5, 3.0],
        'labels': ['1.0', '1.5', '2.0 ★', '2.5', '3.0'],
        'baseline_idx': 2,
        'cmap': plt.cm.Purples,
        'accent': '#9C27B0',
    },
}


def get_t_thresh(c_series, t_series, cv, frac=0.90):
    """Interpolated time to reach frac*c_vessel at neuron surface."""
    cn = np.array(c_series) / cv
    for i in range(len(cn)-1):
        if cn[i] <= frac <= cn[i+1]:
            f = (frac - cn[i]) / (cn[i+1] - cn[i] + 1e-20)
            return t_series[i] + f*(t_series[i+1] - t_series[i])
    return t_series[-1] if cn[-1] >= frac else np.nan


print("\nRunning sensitivity analysis (20 simulations)...")
sens = {}
total = sum(len(v['values']) for v in param_sweeps.values())
run_n = 0

for pname, pinfo in param_sweeps.items():
    sens[pname] = []
    for val in pinfo['values']:
        kwargs = dict(R_v_p=R_v_base, D_ECS_p=D_ECS_base,
                      K_bt_p=K_bt_base, n_hill_p=n_hill_base,
                      t_end=120.0, t_saves=t_saves_s)
        kwargs[pinfo['key']] = val
        r_s, snaps_s, cv_s, idx_s = run_sim(**kwargs)
        c_n = [snaps_s.get(ts, np.zeros(len(r_s)))[idx_s] for ts in t_saves_s]
        t90 = get_t_thresh(c_n, t_saves_s, cv_s, 0.90)
        t50 = get_t_thresh(c_n, t_saves_s, cv_s, 0.50)
        sens[pname].append({'val': val, 'c_n': c_n, 'cv': cv_s,
                             't90': t90, 't50': t50})
        run_n += 1
        print(f"  {run_n}/{total}  {pname.split()[0]}={val:.1e}  "
              f"t50={t50:.1f}s  t90={t90:.1f}s")

print("Done.\n")


# ── FIGURE 2 ──────────────────────────────────────────────────────────────────

fig2 = plt.figure(figsize=(17, 13))
gs2  = gridspec.GridSpec(3, 4, figure=fig2,
                          height_ratios=[1.1, 1.1, 0.95],
                          hspace=0.58, wspace=0.42)

ACCENT = '#f0f0f0'
BG     = '#fafafa'
fig2.patch.set_facecolor(BG)


# ── Row 0: Tornado (full width) ─────────────────────────────────────────────
ax_tor = fig2.add_subplot(gs2[0, :])

tornado_rows = []
for pname, pinfo in param_sweeps.items():
    bl   = sens[pname][pinfo['baseline_idx']]['t90']
    lo   = sens[pname][0]['t90']
    hi   = sens[pname][-1]['t90']
    p_lo = (lo - bl) / bl * 100 if not np.isnan(lo) else 0
    p_hi = (hi - bl) / bl * 100 if not np.isnan(hi) else 0
    tornado_rows.append({
        'label': pname, 'lo': p_lo, 'hi': p_hi,
        'col': pinfo['accent'], 'bl': bl
    })

tornado_rows.sort(key=lambda x: max(abs(x['lo']), abs(x['hi'])), reverse=True)

for i, row in enumerate(tornado_rows):
    lo, hi = min(row['lo'], row['hi']), max(row['lo'], row['hi'])
    ax_tor.barh(i, hi, left=0, color=row['col'], alpha=0.80, edgecolor='k', lw=0.4, height=0.55)
    ax_tor.barh(i, lo, left=0, color=row['col'], alpha=0.38, edgecolor='k', lw=0.4, height=0.55)
    sign = '+' if row['hi'] >= 0 else ''
    ax_tor.text(hi + 0.8, i, f"{sign}{row['hi']:.1f}%", va='center', fontsize=9)
    sign2 = '+' if row['lo'] >= 0 else ''
    ax_tor.text(lo - 0.8, i, f"{sign2}{row['lo']:.1f}%", va='center', ha='right', fontsize=9)

ax_tor.set_yticks(range(len(tornado_rows)))
ax_tor.set_yticklabels([r['label'] for r in tornado_rows], fontsize=10.5)
ax_tor.axvline(0, color='#333333', lw=1.3)
ax_tor.set_xlabel('% change in t₉₀ from baseline', fontsize=10)
ax_tor.set_title(
    'A  |  Sensitivity Tornado — Δt₉₀ (time-to-90% neuron uptake)\n'
    '       Dark bars: low-end parameter value  ·  Light bars: high-end value  '
    '·  (★) marks baseline',
    fontsize=10.5, fontweight='bold', pad=8)
ax_tor.grid(axis='x', alpha=0.25, lw=0.7)
ax_tor.set_facecolor(BG)


# ── Rows 1–2: 4 time-series + 4 t90 bar charts ──────────────────────────────
letters_ts  = ['B','C','D','E']
letters_bar = ['F','G','H','I']

for col_i, (pname, pinfo) in enumerate(param_sweeps.items()):
    n_vals  = len(pinfo['values'])
    clrs    = pinfo['cmap'](np.linspace(0.35, 0.92, n_vals))
    bl_idx  = pinfo['baseline_idx']

    # ── Time series ──
    ax_ts = fig2.add_subplot(gs2[1, col_i])
    ax_ts.set_facecolor(BG)

    for j, (val, res) in enumerate(zip(pinfo['values'], sens[pname])):
        c_n = np.array(res['c_n']) / res['cv'] * 100
        lw  = 2.4 if j == bl_idx else 1.3
        ls  = '-'  if j == bl_idx else '--'
        ax_ts.plot(t_saves_s, c_n, lw=lw, ls=ls, color=clrs[j],
                   label=pinfo['labels'][j], alpha=0.95 if j==bl_idx else 0.72)

    ax_ts.axhline(90, color='#888888', ls=':', lw=1.1, alpha=0.7, label='90% threshold')
    ax_ts.axhline(50, color='#bbbbbb', ls=':', lw=0.8, alpha=0.5)
    ax_ts.set_xlabel('Time (s)', fontsize=9)
    ax_ts.set_ylabel('c_neuron  (%  of  c_vessel)', fontsize=8.5)
    ax_ts.set_title(f'{letters_ts[col_i]}  |  {pname}',
                    fontsize=10, fontweight='bold')
    ax_ts.legend(fontsize=7, ncol=1, loc='lower right',
                 framealpha=0.7, handlelength=1.5)
    ax_ts.set_ylim([45, 103])
    ax_ts.grid(alpha=0.2, lw=0.6)

    # ── t₉₀ bar chart ──
    ax_bar = fig2.add_subplot(gs2[2, col_i])
    ax_bar.set_facecolor(BG)

    t90s = [r['t90'] for r in sens[pname]]
    bl_t90 = t90s[bl_idx]

    bar_clrs = []
    for j, t in enumerate(t90s):
        if j == bl_idx:
            bar_clrs.append('#2ecc71')        # baseline — green
        elif not np.isnan(t) and abs(t - bl_t90)/bl_t90 > 0.08:
            bar_clrs.append(pinfo['accent'])  # significant change
        else:
            bar_clrs.append('#aaaaaa')        # minor change

    bars = ax_bar.bar(range(n_vals), t90s, color=bar_clrs,
                      alpha=0.85, edgecolor='k', lw=0.4, width=0.6)
    ax_bar.axhline(bl_t90, color='#2ecc71', ls='--', lw=1.2, alpha=0.7)
    ax_bar.set_xticks(range(n_vals))
    ax_bar.set_xticklabels(pinfo['labels'], fontsize=7.5, rotation=12)
    ax_bar.set_ylabel('t₉₀  (s)', fontsize=9)
    ax_bar.set_title(f'{letters_bar[col_i]}  |  t₉₀  vs  {pname.split()[0]}',
                     fontsize=10, fontweight='bold')
    ax_bar.grid(axis='y', alpha=0.2, lw=0.6)

    for bar, t in zip(bars, t90s):
        if not np.isnan(t):
            ax_bar.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() + 0.4,
                        f'{t:.0f}s', ha='center', va='bottom', fontsize=7.5)

fig2.suptitle(
    'Isoflurane Diffusion — Sensitivity Analysis\n'
    'Baseline: D_ECS = 3×10⁻¹⁰ m²/s  ·  R_v = 5 μm  ·  '
    'K_bt = 1.4  ·  n_Hill = 2.0    (★ = baseline value)',
    fontsize=12, fontweight='bold', y=1.01)

out2 = '/mnt/user-data/outputs/isoflurane_sensitivity.png'
fig2.savefig(out2, dpi=160, bbox_inches='tight', facecolor=BG)
plt.close(fig2)
print(f"Figure 2 saved: {out2}")
