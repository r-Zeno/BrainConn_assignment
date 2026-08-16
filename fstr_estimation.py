import numpy as np

TWO_PI = 2.0 * np.pi

def epsp_kappa(tau_rise_s, tau_decay_s):
    r, d = float(tau_rise_s), float(tau_decay_s)
    if r <= 0.0:
        return 1.0
    if abs(d - r) < 1e-12:
        return float(np.e)
    tp = (r * d / (d - r)) * np.log(d / r)
    return 1.0 / ((d / (d - r)) * (np.exp(-tp / d) - np.exp(-tp / r)))


def c_hat_from_gain(omega_e, omega_i, G_V, nu0_Hz):
    return 0.5 * (omega_e + omega_i) * G_V / nu0_Hz


def c_hat_from_supra(omega_tilde, dV_supra_mV):
    return omega_tilde / (2.0 * dV_supra_mV)


def omega_tildes(omega_e, omega_i, k, ek):
    return 0.5 * (omega_e + omega_i), omega_e + ek - k, omega_i + k - ek

def mv_to_fstr_curve(omega_f, mV, model,
                     omega_e=1.5, omega_i=0.5, k=0.5, ek=0.0,
                     nu0_Hz=40.0, G_V=1.0, dV_supra_mV=None,
                     tau_m_s=0.010, nu_meas_Hz=40.0,
                     # model 2 only
                     protocol="constant_current",
                     # model 1 only
                     tau_syn_s=0.002, N=1, epsp_area_mV_s=None,
                     correct_summation=False,
                     depression_tau_rec_s=None, depression_U=0.5):

    omega_f = np.atleast_1d(np.asarray(omega_f, dtype=np.float64))
    Omega, wE, wI = omega_tildes(omega_e, omega_i, k, ek)

    if dV_supra_mV is not None:
        c_hat = c_hat_from_supra(wE if model == "model1" else Omega, dV_supra_mV)
    else:
        c_hat = c_hat_from_gain(omega_e, omega_i, G_V, nu0_Hz)

    nu_f = nu0_Hz * omega_f / Omega                      # Hz
    info = {"c_hat": c_hat, "Omega": Omega, "omega_tilde_E": wE,
            "omega_tilde_I": wI, "nu_f": nu_f, "model": model}

    if model == "model2":
        S1 = mV * np.sqrt(1.0 + (TWO_PI * nu_meas_Hz * tau_m_s) ** 2)
        if protocol == "constant_voltage":
            S1 = S1 * (np.sqrt(1.0 + (TWO_PI * nu_f * tau_m_s) ** 2)
                       / np.sqrt(1.0 + (TWO_PI * nu_meas_Hz * tau_m_s) ** 2))
        F = 0.5 * c_hat * np.broadcast_to(S1, nu_f.shape).astype(np.float64)
        u0 = np.zeros_like(F)
        info.update(S1_dc_mV=np.mean(S1), protocol=protocol)

    elif model == "model1":
        if epsp_area_mV_s is None:
            kap = epsp_kappa(tau_syn_s, tau_m_s)
            area = kap * mV * tau_m_s
            if correct_summation:
                area *= (1.0 - np.exp(-1.0 / (nu_meas_Hz * tau_m_s)))
            info["kappa"] = kap
        else:
            area = float(epsp_area_mV_s)
        A = np.full_like(nu_f, area)
        if depression_tau_rec_s:
            U, tr = depression_U, depression_tau_rec_s
            A = area * (1.0 + U * nu_meas_Hz * tr) / (1.0 + U * nu_f * tr)
        s_hat = 1.0 / np.sqrt(1.0 + (TWO_PI * nu_f * tau_syn_s) ** 2)
        u0 = c_hat * N * nu_f * A
        F = u0 * s_hat
        info.update(area_mV_s=area, s_hat=s_hat, r_match=2.0 * s_hat - 1.0)

    else:
        raise ValueError("model does not exist")

    info["u0"] = u0
    info["omega_e_offset"] = u0 - F
    return np.column_stack([F, omega_f]), info


def model_from_modeltype(modeltype):
    return "model1" if int(modeltype) in (0, 1) else "model2"


######## overlay
def overlay_mv_line(ax, curves, ylim=None, colors_=("w", "#ffe08a", "#8ae0ff"),
                    legend=False, scale_x=1.0, scale_y=1.0):
    try:
        import matplotlib.patheffects as pe
        halo = [pe.withStroke(linewidth=2.6, foreground="k")]
    except Exception:
        halo = None
    lo, hi = ylim if ylim is not None else ax.get_ylim()
    drawn = False
    for j, cv in enumerate(curves):
        pts = cv["points"]
        F, wf = pts[:, 0] / scale_y, pts[:, 1] / scale_x
        if np.all(F < lo) or np.all(F > hi):
            continue
        ax.plot(wf, F, lw=1.4, ls="--", color=colors_[j % len(colors_)],
                path_effects=halo, label=cv.get("label"), zorder=6)
        drawn = True
    if legend and drawn:
        ax.legend(loc="upper left", fontsize=7, framealpha=0.75)
    return drawn


def add_gamma_axes(ax, gamma, xlabel=r"$\omega_f/\gamma$", ylabel=r"$f_{str}/\gamma$",
                   pad=0.0):
    if not gamma:
        return None, None
    fwd, inv = (lambda v: v / gamma), (lambda v: v * gamma)
    sx = ax.secondary_xaxis("top", functions=(fwd, inv))
    sy = ax.secondary_yaxis(1.0 + pad, functions=(fwd, inv))
    sx.set_xlabel(xlabel, fontsize=8, labelpad=1)
    sy.set_ylabel(ylabel, fontsize=8, labelpad=1)
    sx.tick_params(labelsize=7)
    sy.tick_params(labelsize=7)
    return sx, sy