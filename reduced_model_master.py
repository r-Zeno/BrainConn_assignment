import os
os.environ["PYTHON_JULIACALL_THREADS"] = "auto"
os.environ["PYTHON_JULIACALL_HANDLE_SIGNALS"] = "yes"
from matplotlib import pyplot as plt
from matplotlib import cm
from juliacall import Main as jl
import matplotlib.colors as colors
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import gc
from fstr_estimation import (mv_to_fstr_curve, model_from_modeltype,
                              overlay_mv_line, add_gamma_axes)

TONGUE_RATIOS = [(1, 1), (1, 2), (1, 3), (2, 3)]
TONGUE_COLORS = ["#cccccc", "#e41a1c", "#4daf4a", "#984ea3", "#f781bf"]

# which populations par_sweep returns
GROUP_LABELS = {1: ("E",), 2: ("I",), 3: ("E", "I")}

def classify_tongues(winding_map, tol=0.005):
    min_dev = np.full(winding_map.shape, np.inf, dtype=np.float64)
    tongue_id = np.zeros(winding_map.shape, dtype=np.int16)
    for i, (p, q) in enumerate(TONGUE_RATIOS):
        dev = np.abs(winding_map - p/q)
        better = dev < min_dev
        min_dev = np.where(better, dev, min_dev)
        tongue_id = np.where(better, i+1, tongue_id).astype(np.int16)

    tongue_id = np.where(min_dev < tol, tongue_id, 0).astype(np.int16)
    return tongue_id

def compute_strobe_coherence(xE, yE, t_settled, omega_f_arr):
    y_len, x_len, T = xE.shape
    strobe_coherence = np.zeros((y_len, x_len), dtype=np.float32)
    for xi, om_f in enumerate(omega_f_arr):
        T_f = 2.0 * np.pi / float(om_f)
        # multiples of forcing period
        n0, n1 = int(np.ceil(float(t_settled[0]) / T_f)), int(float(t_settled[-1]) / T_f)
        strobes = T_f * np.arange(n0, n1 + 1)
        strobes = strobes[(strobes >= t_settled[0]) & (strobes <= t_settled[-1])]
        if len(strobes) < 3: continue
        
        idx = np.searchsorted(t_settled, strobes).clip(0, T - 1)

        z = (xE[:, xi, :][:, idx] + 1j * yE[:, xi, :][:, idx]).astype(np.complex64)
        z_abs = np.abs(z)
        v = z_abs > 1e-7
        z_norm = np.where(v, z / np.where(v, z_abs, 1.0), 0.0 + 0.0j)
        strobe_coherence[:, xi] = np.abs(np.mean(z_norm, axis=-1))
    return strobe_coherence

def analyse_group(sol_re, sol_comp, timesteps, x, chunk_size=50):
    measures = {}

    ############ last order
    measures["last_re"] = sol_re[:, :, -1]

    ############ error
    settle4mean = int(len(timesteps)/2)
    settle_mean = np.mean(sol_re[:, :, -settle4mean:], axis=-1, keepdims=True)
    measures["sse"] = np.sum(timesteps*np.abs(sol_re - settle_mean), axis=-1)

    measures["alt_sse"] = np.sum(np.sqrt((sol_re - sol_re[:, :, -1:])**2), axis=-1)

    ############ mean freq, rotation number, and coherence
    mask = timesteps > int((max(timesteps)/2))
    time_freq = timesteps[mask]
    time_delta = time_freq[-1] - time_freq[0]

    y_len, x_len, T = sol_comp[0].shape
    phase_delta = np.zeros((y_len, x_len), dtype=np.float32)

    for y0 in range(0, y_len, chunk_size):
        y1 = min(y0 + chunk_size, y_len)
        # float64 to avoid numerical precision loss during unwrap? does this make sense? ai says so
        chunk_phase_rad = np.arctan2(sol_comp[1, y0:y1, :, :].astype(np.float64), 
                                     sol_comp[0, y0:y1, :, :].astype(np.float64))
        chunk_phase = np.unwrap(chunk_phase_rad, axis=-1)
        phase_delta[y0:y1, :] = (chunk_phase[:, :, -1] - chunk_phase[:, :, 0]).astype(np.float32)
        
        del chunk_phase_rad, chunk_phase

    measures["mean_f"] = (phase_delta / time_delta) + x[np.newaxis, :]

    # arnold tongues
    measures["winding"] = measures["mean_f"] / x[np.newaxis, :]
    measures["tongue_id"] = classify_tongues(measures["winding"])
    measures["strobe_coh"] = compute_strobe_coherence(sol_comp[0], sol_comp[1], time_freq, x)

    return measures

def run_sweep(x, y, timesteps, paras, modeltype, group_type):
    sol = jl.par_sweep(x, y, timesteps, paras, modeltype, group_type)

    results = {}
    for i, group in enumerate(GROUP_LABELS[group_type]):
        sol_re_raw = np.array(sol[2*i], dtype=np.float32, copy=False)
        sol_comp_stable_raw = np.array(sol[2*i + 1], dtype=np.float32, copy=False)

        sol_re = sol_re_raw.transpose(2, 1, 0)
        # for stroboscopic coherence
        sol_comp_stable = sol_comp_stable_raw.transpose(0, 3, 2, 1)

        results[group] = analyse_group(sol_re, sol_comp_stable, timesteps, x)

        del sol_re_raw, sol_comp_stable_raw, sol_re, sol_comp_stable
        gc.collect()

    del sol
    gc.collect()
    return results

def plot_group(measures, x, y, k_stim, group, save_dir, mv_curves=None,
               gamma=None, bif_boundaries=None):
    ynorm = float(gamma) if gamma else 1.0
    xs, ys = np.asarray(x), np.asarray(y)/ynorm
    ylim = (float(ys[0]), float(ys[-1]))
    xlab = r"Forcing Frequency $\left(\omega_f/Omega\right)$"
    ylab = r"Forcing Strength $\left(F/\gamma\right)$"

    cb_pad = 0.05

    def _decorate(ax_, legend=True):
        if mv_curves:
            overlay_mv_line(ax_, mv_curves, ylim=ylim, legend=legend,
                            scale_x=1.0, scale_y=ynorm)
    fig, ax = plt.subplots()
    mesh = ax.pcolormesh(xs, ys, measures["last_re"])
    fig.colorbar(mesh, ax=ax, label = f"{group} group order", pad=cb_pad)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    _decorate(ax)
    plot_n = f"k{k_stim}_re_{group}.png"
    fig.savefig(os.path.join(save_dir, plot_n), dpi=1200)

    sse = measures["sse"]
    sse_pos = sse[sse > 0]
    sse_norm = colors.LogNorm(vmin=sse_pos.min(), vmax=sse.max()) if sse_pos.size else None
    fig1, ax1 = plt.subplots()
    mesh1 = ax1.pcolormesh(xs, ys, sse, norm = sse_norm, cmap="viridis")
    fig1.colorbar(mesh1, ax=ax1, label = f"SSE {group}", pad=cb_pad)
    ax1.set_xlabel(xlab)
    ax1.set_ylabel(ylab)
    _decorate(ax1)
    plot_n1 = f"k{k_stim}_sse_{group}.png"
    fig1.savefig(os.path.join(save_dir, plot_n1), dpi = 600)

    fig3, ax3 = plt.subplots()
    mesh3 = ax3.pcolormesh(xs, ys, measures["mean_f"])
    fig3.colorbar(mesh3, ax=ax3, label=rf"average frequency $(\omega_f/\Omega)$, {group} group, k={k_stim}", pad=cb_pad)
    # overlay bif
    if bif_boundaries is not None:
        for b in bif_boundaries:
            ax3.plot(b[:, 0], b[:, 1]/ynorm, color='black', linewidth=2.0, linestyle='-')
    ax3.set_xlabel(xlab)
    ax3.set_ylabel(ylab)
    _decorate(ax3)
    plot_n3 = f"k{k_stim}_obsfreq_{group}.png"
    fig3.savefig(os.path.join(save_dir, plot_n3), dpi=600)

    ############ Plotting: Winding Number
    fig_w, ax_w = plt.subplots()
    w_clim = np.nanpercentile(measures["winding"], [2, 98])
    im_w = ax_w.pcolormesh(xs, ys, measures["winding"], cmap="RdBu_r", vmin=w_clim[0], vmax=w_clim[1])
    fig_w.colorbar(im_w, ax=ax_w, label="Winding Number (W = f_obs / ω_f)", pad=cb_pad)
    ax_w.set_title(f"Winding Number ({group}, k={k_stim})")
    ax_w.set_xlabel(xlab)
    ax_w.set_ylabel(ylab)
    _decorate(ax_w, legend=False)
    fig_w.tight_layout()
    fig_w.savefig(os.path.join(save_dir, f"k{k_stim}_winding_number_{group}.png"), dpi=600)

    ############ Plotting: Arnold Tongues
    fig_t, ax_t = plt.subplots()
    cmap_t = colors.ListedColormap(TONGUE_COLORS[:len(TONGUE_RATIOS) + 1])
    bnds = np.arange(-0.5, len(TONGUE_RATIOS) + 1.5)
    norm_t = colors.BoundaryNorm(bnds, cmap_t.N)
    mesh_t = ax_t.pcolormesh(xs, ys, measures["tongue_id"], cmap=cmap_t, norm=norm_t)

    patches = [mpatches.Patch(color=TONGUE_COLORS[0], label="unlocked")]
    patches += [mpatches.Patch(color=TONGUE_COLORS[i + 1], label=f"{p}:{q}") for i, (p, q) in enumerate(TONGUE_RATIOS)]
    ax_t.legend(handles=patches, loc="upper right", fontsize=8, ncol=2, framealpha=0.8)
    ax_t.set_title(f"Phase lock ratios ({group}, k={k_stim})")
    ax_t.set_xlabel(xlab)
    ax_t.set_ylabel(ylab)
    _decorate(ax_t, legend=False)
    fig_t.tight_layout()
    fig_t.savefig(os.path.join(save_dir, f"k{k_stim}_arnold_tongues_{group}.png"), dpi=600)

    # Stroboscopic coherence
    fig_coh, ax_coh = plt.subplots()
    mesh_coh = ax_coh.pcolormesh(xs, ys, measures["strobe_coh"], cmap="magma", vmin=0, vmax=1)
    fig_coh.colorbar(mesh_coh, ax=ax_coh, label="Stroboscopic Coherence", pad=cb_pad)
    ax_coh.set_title(f"Stroboscopic Coherence ({group}, k={k_stim})\n(1.0 = Perfectly Locked)")
    ax_coh.set_xlabel(xlab)
    ax_coh.set_ylabel(ylab)
    _decorate(ax_coh)
    fig_coh.savefig(os.path.join(save_dir, f"k{k_stim}_coherence_{group}.png"), dpi=600)

def plot_order_surface(measures, x, y, k_stim, group, save_dir, max_cells=200,
                       cmap="turbo", clim_pct=(2, 98), view=(22, -120),
                       shade_contrast=0.55, iso_levels=False, mesh_edges=False,
                       floor_shadow=False, box_aspect=(1, 1, 0.5),
                       gamma=None, mv_curves=None):

    order = measures["last_re"]
    freq = measures["mean_f"]

    y_len, x_len = order.shape
    sy = max(1, int(np.ceil(y_len / max_cells)))
    sx = max(1, int(np.ceil(x_len / max_cells)))
    ynorm = float(gamma) if gamma else 1.0
    xs = np.asarray(x[::sx])
    ys_raw = np.asarray(y[::sy])
    ys = ys_raw / ynorm
    Z = order[::sy, ::sx]
    C = freq[::sy, ::sx]
    X, Y = np.meshgrid(xs, ys)
    zmin, zmax = float(np.nanmin(Z)), float(np.nanmax(Z))

    # percentile clip for color range
    vmin, vmax = np.nanpercentile(C, clim_pct)
    norm_c = colors.Normalize(vmin=vmin, vmax=vmax)
    cmap_c = plt.get_cmap(cmap)
    rgb = cmap_c(norm_c(C))[..., :3]

    if shade_contrast:
        light = colors.LightSource(azdeg=315, altdeg=45)
        inten = light.hillshade(Z, vert_exag=1.0, dx=float(xs[1] - xs[0]),
                                dy=float(ys_raw[1] - ys_raw[0]))
        rgb = np.clip(rgb * (1 - shade_contrast + 2*shade_contrast*inten[..., None]), 0, 1)

    if iso_levels:
        band = (Z - zmin) / max(zmax - zmin, 1e-9) * iso_levels
        on_line = np.abs(band - np.round(band)) < 0.07
        rgb = np.where(on_line[..., None], rgb*0.55, rgb)

    facecolors = np.concatenate([rgb, np.ones(rgb.shape[:2] + (1,))], axis=-1)

    fig_s = plt.figure(figsize=(8, 6))
    ax_s = fig_s.add_subplot(111, projection="3d")
    surf = ax_s.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=facecolors,
                             shade=False, antialiased=False, rasterized=True)
    if mesh_edges:
        surf.set_edgecolor((0, 0, 0, 0.22))
        surf.set_linewidth(0.12)
    else:
        surf.set_linewidth(0)

    z_lo = zmin - 0.3*(zmax - zmin) if floor_shadow else zmin
    if floor_shadow:
        ax_s.contourf(X, Y, Z, zdir="z", offset=z_lo, levels=12, cmap="Greys", alpha=0.55)

    if mv_curves and len(xs) >= 2 and len(ys_raw) >= 2:
        def _z_on_surface(xq, yq):
                    ix = np.clip(np.searchsorted(xs, xq) - 1, 0, len(xs) - 2)
                    iy = np.clip(np.searchsorted(ys_raw, yq) - 1, 0, len(ys_raw) - 2)
                    tx = (xq - xs[ix]) / (xs[ix + 1] - xs[ix])
                    ty = (yq - ys_raw[iy]) / (ys_raw[iy + 1] - ys_raw[iy])
                    z0 = (1 - tx)*Z[iy, ix]     + tx*Z[iy, ix + 1]
                    z1 = (1 - tx)*Z[iy + 1, ix] + tx*Z[iy + 1, ix + 1]
                    return (1 - ty)*z0 + ty*z1
    
        lift = 0.015*max(zmax - zmin, 1e-9)
        for cv in mv_curves:
            pts = np.asarray(cv["points"])
            F_raw, wf = pts[:, 0], pts[:, 1]
            keep = ((wf >= xs[0]) & (wf <= xs[-1]) &
                    (F_raw >= ys_raw[0]) & (F_raw <= ys_raw[-1]))
            if not np.any(keep):
                continue
            wf_k, F_k = wf[keep], F_raw[keep]
            zk = _z_on_surface(wf_k, F_k) + lift
            ax_s.plot(wf_k, F_k/ynorm, zk, lw=1.8, ls="--",
                      color="0.35", zorder=10)

    mappable = cm.ScalarMappable(norm=norm_c, cmap=cmap_c)
    mappable.set_array(C)
    fig_s.colorbar(mappable, ax=ax_s, shrink=0.6, pad=0.1,
                   label=r"mean settle freq $(\omega_f/\Omega)$, E group")

    ax_s.set_xlabel("Forcing Frequency")
    ax_s.set_ylabel("Forcing Strength")
    ax_s.set_zlabel(f"{group} group order")
    ax_s.set_zlim(z_lo, zmax)
    ax_s.set_box_aspect(box_aspect)
    ax_s.view_init(elev=view[0], azim=view[1])

    fig_s.savefig(os.path.join(save_dir, f"k{k_stim}_order_surface_{group}.png"), dpi=600)

def plot_delta(measures_high, measures_low, x, y, group, save_dir):
    """High vs low k comparison, one set per group."""
    delta_sse = measures_high["sse"] - measures_low["sse"]
    alt_delta_sse = measures_high["alt_sse"] - measures_low["alt_sse"]

    plt.figure()
    plt.hist(delta_sse.flatten(), bins=100)
    plt.yscale('log')
    plt.xlabel('delta SSE')
    plt.ylabel('count')
    plt.title(f"{group} group")

    delta_masked = np.where(delta_sse > 0, delta_sse, np.nan) # make negative delta nan for now, useful to have nicer gradients in the plot
    vmin = np.nanpercentile(delta_masked, 1)
    vmax = np.nanpercentile(delta_masked, 99)

    alt_delta_masked = np.where(alt_delta_sse > 0, alt_delta_sse, np.nan) # make negative delta nan for now
    alt_vmin = np.nanpercentile(alt_delta_masked, 1)
    alt_vmax = np.nanpercentile(alt_delta_masked, 99)

    fig2, ax2 = plt.subplots()
    cmap = plt.cm.viridis.copy()
    cmap.set_bad('lightgrey')
    mesh2 = ax2.pcolormesh(x, y, alt_delta_sse, norm=colors.LogNorm(vmin=alt_vmin, vmax=alt_vmax), cmap=cmap)
    fig2.colorbar(mesh2, ax=ax2, label = "delta SSE")
    ax2.set_title(f"{group} group")
    fig2.savefig(os.path.join(save_dir, f"delta_SSE_{group}.png"), dpi = 1200)

    max_dsse = np.max(np.abs(delta_sse))
    min_dsse = -max_dsse
    symlog = colors.SymLogNorm(linthresh=1e-3, vmin=min_dsse, vmax=max_dsse, base=10)
    fig5, ax5 = plt.subplots()
    mesh5 = ax5.pcolormesh(x, y, delta_sse, norm=symlog, cmap="bwr")
    fig5.colorbar(mesh5, ax=ax5, label="delta SSE (symlog)")
    ax5.set_title(f"{group} group")
    fig5.savefig(os.path.join(save_dir, f"delta_SSE_symlog_{group}.png"), dpi = 600)

    return delta_sse, alt_delta_sse

print(f"Julia is running with {jl.Threads.nthreads()} threads")
curr_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
save_dir = os.path.join(curr_dir, "plots")
os.makedirs(save_dir, exist_ok=True)

code_dir = os.path.dirname(os.path.abspath(__file__))
jl_code = os.path.join(code_dir, "gpusol.jl")
jl.include(jl_code)

sysan_code = os.path.join(code_dir, "sysan.jl")
jl.include(sysan_code)

lower_k = False

############ mV -> f_str calibration
SHOW_MV_LINE = True

# which stimulation the f_str axis represents. auto decides based on the selected model | cos | sin
STIM_MODEL = "auto"

MV_VALUES = [2.0]

MV_NU0_HZ   = 20.0    # arb rhythm Hz
MV_NU_MEAS  = 40.0    # frequency at which the mV value was measured
MV_G_V      = 1.0     # f-I slpe
MV_DV_SUPRA = None
MV_TAU_M    = 0.010
MV_TAU_SYN  = 0.0
MV_N        = 2       # trains per volley
MV_AREA     = None
MV_SUMMATION      = True
MV_DEPRESSION_TAU = None
MV_PROTOCOL = "constant_current"

# 0 = asymmetric cos | 1 = symmetric cos | 2 = symmetric sin | 3 = asymmetric sin
modeltype = 1

# which populations are saved, returned and plotted
# 1 = E only | 2 = I only | 3 = both
group_type = 1

if group_type not in GROUP_LABELS:
    raise ValueError("group_type must be 1 (E only), 2 (I only) or 3 (both)")

# Parameters
omega_e = 1.5
omega_i = 0.5
omega_f = 2
f_str = 0.7
ek = 0.0
k_init = 0.5
k_stim_high = 0.5
k_stim_low = 0.2
gamma = 0.1
f_init = 0.0
grid_size = 750
T = 500

paras_high = np.array([omega_e, omega_i, omega_f, f_init, ek, k_init, k_stim_high, gamma])
paras_low = np.array([omega_e, omega_i, omega_f, f_init, ek, k_init, k_stim_low, gamma])

x = np.linspace(1, 3.5, grid_size, dtype=np.float32)
y = np.linspace(0, 1, grid_size, dtype=np.float32)
timesteps = np.linspace(0, T, 2500, dtype=np.float32)


############ F curves
mv_curves = None
if SHOW_MV_LINE:
    _model = model_from_modeltype(modeltype) if STIM_MODEL == "auto" else STIM_MODEL
    mv_curves = []
    for _mv in MV_VALUES:
        _pts, _info = mv_to_fstr_curve(
            x, _mv, _model,
            omega_e=omega_e, omega_i=omega_i, k=k_init, ek=ek,
            nu0_Hz=MV_NU0_HZ, G_V=MV_G_V, dV_supra_mV=MV_DV_SUPRA,
            tau_m_s=MV_TAU_M, nu_meas_Hz=MV_NU_MEAS, protocol=MV_PROTOCOL,
            tau_syn_s=MV_TAU_SYN, N=MV_N, epsp_area_mV_s=MV_AREA,
            correct_summation=MV_SUMMATION,
            depression_tau_rec_s=MV_DEPRESSION_TAU)
        mv_curves.append({"points": _pts, "info": _info, "label": f"{_mv} mV"})
        _F = _pts[:, 0]
        print(f"[mV] {_model} {_mv} mV -> f_str {_F.min():.4f}..{_F.max():.4f}"
              f"  (f_str/gamma {_F.min()/gamma:.2f}..{_F.max()/gamma:.2f})")
        if _model == "model1":
            _off = _info["omega_e_offset"]
            print(f"     add {_off.min():.4f}..{_off.max():.4f} to omega_e to keep the"
                  f" DC shift exact (r=1 forces u0 = f_str)")
        if _F.max() < y[0] or _F.min() > y[-1]:
            print(f"     WARNING: {_mv} mV lies outside the f_str axis [{y[0]:g}, {y[-1]:g}]")

###### Bifurcation
print("Bif in Julia start...")
bif_model_choice = "cos" if modeltype in [0, 1] else "sin"
raw_boundaries = jl.get_entrainment_boundaries(bif_model_choice)

x_min, x_max = float(x[0]), float(x[-1])
y_min, y_max = float(y[0]), float(y[-1])
filtered_boundaries = []

for b in raw_boundaries:
    b_np = np.array(b)
    om_f_vals = b_np[:, 0]
    f_str_vals = b_np[:, 1]
    
    mask = (om_f_vals >= x_min) & (om_f_vals <= x_max) & (f_str_vals >= y_min) & (f_str_vals <= y_max)
    
    if np.any(mask):
        om_filtered = np.where(mask, om_f_vals, np.nan)
        f_str_filtered = np.where(mask, f_str_vals, np.nan)
        filtered_boundaries.append(np.column_stack((om_filtered, f_str_filtered)))

start = time.time()

measures_high = run_sweep(x, y, timesteps, paras_high, modeltype, group_type)

if lower_k:
    measures_low = run_sweep(x, y, timesteps, paras_low, modeltype, group_type)

end = time.time()
print(f"done. took {round(end-start, 2)} secs")

############ plotting
############ high k plots
for group in GROUP_LABELS[group_type]:
    plot_group(measures_high[group], x, y, k_stim_high, group, save_dir, mv_curves,
               gamma=gamma, bif_boundaries=filtered_boundaries)
    plot_order_surface(measures_high[group], x, y, k_stim_high, group, save_dir,
                       gamma=gamma, mv_curves=mv_curves)

############ lowered k plots
if lower_k:
    for group in GROUP_LABELS[group_type]:
        plot_group(measures_low[group], x, y, k_stim_low, group, save_dir, mv_curves,
                   gamma=gamma, bif_boundaries=filtered_boundaries)
        plot_order_surface(measures_low[group], x, y, k_stim_low, group, save_dir,
                           gamma=gamma, mv_curves=mv_curves)

plt.show()