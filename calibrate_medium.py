import numpy as np
from scipy.optimize import minimize
from constants import (
    M_E_MEV,
    R_E_COMPTON_FM,
    R_CORE_VAC_FM,
)


from preprocessor import build_3d_grid, init_hopf_vortex_3d
from solution import relax_vortex_3d
from postprocessor import analyze_relaxed_vortex_3d

DEFAULT_TOPO_SEQUENCE = [
    (1, 2),
    (2, 3),
    (2, 5),
    (3, 4),
    (3, 5),
    (5, 8),
    (7, 11),
    (8, 13),
    (11, 17),
    (13, 21),
    (17, 29),
    (19, 31),
    (21, 34),
    (23, 37),
    (25, 41),
    (27, 43),
    (29, 47),
    (31, 53),
    (33, 55),
    (34, 55),
    (35, 59),
    (37, 73),
    (73, 37),
    (41, 79),
    (47, 89),
    (55, 89),
    (59, 107),
    (61, 113),
    (65, 119),
    (67, 127),
    (71, 131),
    (73, 137),
    (77, 139),
    (79, 143),
    (83, 149),
    (89, 144),
    (91, 163),
    (95, 173),
    (97, 181),
    (101, 187),
    (103, 191),
    (113, 211),
    (144, 233),
]

def evaluate_single_vortex_dynamics(
    m_tor,
    n_pol,
    medium,
    R0_init=R_E_COMPTON_FM,
    N_grid=32,
    dyn_steps=160,
#    max_N=68,
    max_N=68,
):
    N_eff = min(max(int(N_grid), int(2 * max(abs(m_tor), abs(n_pol)))), int(max_N))


    grid = build_3d_grid(R0_init, N=N_eff, box_scale=3.2)

    psi0 = init_hopf_vortex_3d(grid, R0_init, R0_init * 0.30, int(m_tor), int(n_pol))

    res = relax_vortex_3d(
        psi0,
        grid,
        medium,
        max_steps=int(dyn_steps),
        renormalize=False,
        damping_coeff=0.02,
    )
    obs = analyze_relaxed_vortex_3d(res["psi"], grid, medium, int(m_tor), int(n_pol))

    vol_scale = max((float(R0_init) / R_CORE_VAC_FM) ** 2, 1.0)
    mass_scaled = float(obs["mass_mev"]) / vol_scale
    survived = bool(
        res.get("dynamically_stable", False)
        and obs.get("buckling_ok", False)
        and obs["topology"]["topology_preserved"]
    )

    return {
        "topo": (int(m_tor), int(n_pol)),
        "survived": survived,
        "buckling_ok": bool(obs.get("buckling_ok", False)),
        "dynamically_stable": bool(res.get("dynamically_stable", False)),
        "topology_preserved": bool(obs["topology"]["topology_preserved"]),
        "chi_max": float(obs.get("chi_max", 0.0)),
        "mass_mev": mass_scaled,
        "R0_fm": float(obs["R0_fm"]),
        "g_factor": float(obs["g_factor"]),
        "steps_done": int(res["steps_done"]),
    }
def calibrate_single_and_scan_spectrum(
    base_medium,
    base_topo=(2, 3),
    topo_sequence=None,
    N_grid=32,
    dyn_steps=140,
    max_iter=35,
):
    """
    1) Настраивает параметры среды на одном младшем узле base_topo
       с обязательным прохождением buckling и динамической устойчивости.
    2) Последовательно проверяет в этой среде ряд узлов по одному.
    """
    if topo_sequence is None:
        topo_sequence = DEFAULT_TOPO_SEQUENCE

    m_base, n_base = int(base_topo[0]), int(base_topo[1])
    keys = ["alpha_nl", "gamma_nl", "beta_nl", "eta_visc", "delta_pack"]
    x0 = np.log10([max(float(base_medium[k]), 1e-6) for k in keys])

    best_state = {
        "loss": float("inf"),
        "medium": dict(base_medium),
        "base_result": None,
        "spectrum": [],
    }
    def loss_fn(log_params):
        low_b = np.array([-4.0, -3.0, -3.5, -3.0, -4.0])
#        high_b = np.array([1.5, 2.0, 0.5, -0.6, 1.5])
#        high_b = np.array([1.5, 2.0, 0.0, -1.0, 1.5])
        high_b = np.array([1.5, 2.0, 0.0, -1.6, 1.5])


        vals = 10.0 ** np.clip(log_params, low_b, high_b)
        med = dict(base_medium)
        for k, v in zip(keys, vals):
            med[k] = float(v)

        res_single = evaluate_single_vortex_dynamics(
            m_base, n_base, med, R0_init=R_E_COMPTON_FM, N_grid=N_grid, dyn_steps=dyn_steps
        )
        m_est = res_single["mass_mev"]
        if not np.isfinite(m_est) or m_est <= 0:
            return 1e6

        penalty = 0.0
        if not res_single["buckling_ok"]:
            penalty += 50.0 + 10.0 * max(res_single["chi_max"] - 1.0, 0.0)
        if not res_single["dynamically_stable"]:
            penalty += 30.0
        if not res_single["topology_preserved"]:
            penalty += 40.0

        err_m = (np.log10(m_est / M_E_MEV)) ** 2
        err_r = (np.log10(max(res_single["R0_fm"], 1e-4) / R_E_COMPTON_FM)) ** 2
        total_loss = float(err_m + 0.5 * err_r + penalty)

        if total_loss < best_state["loss"]:
            best_state.update(
                {"loss": total_loss, "medium": med, "base_result": res_single}
            )
        return total_loss

  #  minimize(loss_fn, x0, method="Nelder-Mead", options={"maxiter": int(max_iter), "xatol": 1e-3, "fatol": 1e-4})
    minimize(loss_fn, x0, method="Nelder-Mead", options={"maxiter": max(int(max_iter), 60), "xatol": 1e-4, "fatol": 1e-5})

    best_med = best_state["medium"]
    spectrum = []
    for (m_i, n_i) in topo_sequence:
        scale_r = (2.0 * 3.0) / max(float(m_i * n_i), 1.0)
        r0_guess = R_E_COMPTON_FM * scale_r

        item = evaluate_single_vortex_dynamics(
            m_i, n_i, best_med, R0_init=r0_guess, N_grid=N_grid, dyn_steps=dyn_steps
        )
        spectrum.append(item)

    best_state["spectrum"] = spectrum
    return best_state
FOUND_PARTICLES_BENCHMARK = [
    ("Электрон (e⁻)", (2, 3), 0.511, 386.16),
    ("Мюон (μ⁻)", (19, 31), 105.66, 1.87),
    ("Пион (π⁰)", (21, 34), 134.98, 1.46),
    ("Каон (K±)", (27, 43), 493.68, 0.40),
    ("Эта-мезон (η)", (21, 34), 547.86, 0.36),
    ("Ро/Омега (ρ/ω)", (29, 47), 775.26, 0.25),
]


def cross_calibrate_all_particles(
    base_medium,
    particles=None,
    N_grid=32,
    dyn_steps=110,
    max_iter=35,
    progress_callback=None,
):
    if particles is None:
        particles = FOUND_PARTICLES_BENCHMARK
    keys = ["alpha_nl", "gamma_nl", "beta_nl", "eta_visc", "delta_pack"]
    x0 = np.log10([max(float(base_medium[k]), 1e-6) for k in keys])
    results = []
    for idx, (name, (m_p, n_p), m_target, r_target) in enumerate(particles):

#    for name, (m_p, n_p), m_target, r_target in particles:
        scale_r = (2.0 * 3.0) / max(float(m_p * n_p), 1.0)
        r0_guess = R_E_COMPTON_FM * scale_r
        best_p = {"loss": float("inf"), "medium": dict(base_medium), "obs": None}

        def p_loss(log_params):
            low_b = np.array([-4.0, -3.0, -3.5, -3.0, -4.0])
            high_b = np.array([1.5, 2.0, 0.0, -1.6, 1.5])
            vals = 10.0 ** np.clip(log_params, low_b, high_b)
            med = dict(base_medium)
            for k, v in zip(keys, vals):
                med[k] = float(v)
            res_p = evaluate_single_vortex_dynamics(
                m_p, n_p, med, R0_init=r0_guess, N_grid=N_grid, dyn_steps=dyn_steps, max_N=40
            )

            m_est = res_p["mass_mev"]
            if not np.isfinite(m_est) or m_est <= 0:
                return 1e6
            pen = 0.0 if res_p["buckling_ok"] else 50.0
            pen += 0.0 if res_p["dynamically_stable"] else 30.0
            err_m = (np.log10(m_est / float(m_target))) ** 2
            err_r = (np.log10(max(res_p["R0_fm"], 1e-4) / float(r_target))) ** 2
            err = err_m + 0.5 * err_r + pen
            if err < best_p["loss"]:
                best_p.update({"loss": float(err), "medium": med, "obs": res_p})
            return float(err)

        minimize(
            p_loss,
            x0,
            method="Nelder-Mead",
            options={"maxiter": max(int(max_iter), 65), "xatol": 1e-4, "fatol": 1e-5},
        )

    
        m_found = best_p["obs"]["mass_mev"] if best_p["obs"] else 0.0
        err_pct = abs(m_found - float(m_target)) / float(m_target) * 100.0
        med_f = best_p["medium"]
        results.append(
            {
                "Эталонная частица": name,
                "Топология": f"({m_p}, {n_p})",
                "Масса табл. (МэВ)": round(float(m_target), 4),
                "Масса выч. (МэВ)": round(float(m_found), 4),
                "Погр. массы (%)": round(float(err_pct), 2),
                "alpha_nl": round(med_f["alpha_nl"], 6),
                "gamma_nl": round(med_f["gamma_nl"], 5),
                "beta_nl": round(med_f["beta_nl"], 4),
                "eta_visc": round(med_f["eta_visc"], 5),
                "delta_pack": round(med_f["delta_pack"], 4),
            }
        )
        if progress_callback is not None:
            progress_callback(idx + 1, len(particles), name, results[-1])

    return results
