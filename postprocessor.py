import numpy as np
from constants import ALPHA_FS, HBAR_C_MEV_FM
from matrix_assembly import _grad_3d, compute_medium_fields, evaluate_energy_and_gradient


def measure_winding_numbers_3d(psi, grid, R0_est_fm, m_target, n_target):
    """
    Тест №2: измеряет реальные топологические числа намотки (m_tor, n_pol)
    релаксированного 3D-поля psi по циркуляции фазы и проверяет сохранение узла.
    """
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    dx = grid["dx"]
    u = np.real(psi * np.conj(psi))
    u_safe = np.maximum(u, 1e-16)

    gx, gy, gz = _grad_3d(psi, dx)
    # Локальный градиент фазы grad(phi) = Im(psi* grad psi) / |psi|^2
    vx = np.zeros_like(u)
    vy = np.zeros_like(u)
    vz = np.zeros_like(u)
    vx[1:-1, :, :] = np.angle(psi[2:, :, :] * np.conj(psi[:-2, :, :])) * (0.5 / dx)
    vy[:, 1:-1, :] = np.angle(psi[:, 2:, :] * np.conj(psi[:, :-2, :])) * (0.5 / dx)
    vz[:, :, 1:-1] = np.angle(psi[:, :, 2:] * np.conj(psi[:, :, :-2])) * (0.5 / dx)


    
#    vx = np.imag(np.conj(psi) * gx) / u_safe
 #   vy = np.imag(np.conj(psi) * gy) / u_safe
 #   vz = np.imag(np.conj(psi) * gz) / u_safe

    rho = np.sqrt(X**2 + Y**2) + 1e-12
    # Азимутальная скорость фазы вокруг оси Z: d(phase)/d(phi) = -Y*vx + X*vy
    m_local = -Y * vx + X * vy

    # Полоидальная скорость фазы вокруг осевой окружности тора радиуса R0_est_fm
    v_rho = (X * vx + Y * vy) / rho
    d_rho = rho - float(R0_est_fm)
    n_local = d_rho * vz - Z * v_rho

    # Взвешиваем по плотной части силовой трубки (где |psi|^2 > 15% от максимума)
#    mask_core = u > (0.15 * np.max(u))
    mask_core = u > (0.45 * np.max(u))
    alias_scale = float(grid.get("alias_scale", 1.0))
    if np.any(mask_core):
        weights = u[mask_core]
        w_sum = float(np.sum(weights))
        m_meas = float(np.sum(m_local[mask_core] * weights) / w_sum) * alias_scale
        n_meas = float(np.sum(n_local[mask_core] * weights) / w_sum) * alias_scale
    else:
        m_meas = 0.0
        n_meas = 0.0
    m_t, n_t = int(m_target), int(n_target)
#    m_ok = (abs(m_meas - m_t) <= max(0.8, 0.45 * abs(m_t))) and (m_meas * m_t >= 0)
#    n_ok = (abs(n_meas - n_t) <= max(0.8, 0.45 * abs(n_t))) and (n_meas * n_t >= 0)
#    m_ok = (m_meas >= 0.35 * abs(m_t)) and (m_meas <= 1.5 * abs(m_t)) and (m_meas * m_t >= 0)
#    n_ok = (n_meas >= 0.35 * abs(n_t)) and (n_meas <= 1.5 * abs(n_t)) and (n_meas * n_t >= 0)
#    m_ok = (m_meas >= 0.35 * abs(m_t)) and (m_meas <= 1.5 * abs(m_t)) and (m_meas * m_t >= 0)
#    n_ok = (n_meas >= 0.05 * abs(n_t)) and (n_meas <= 1.5 * abs(n_t)) and (n_meas * n_t >= 0)
#    m_ok = (m_meas >= 0.12 * abs(m_t)) and (m_meas <= 1.5 * abs(m_t)) and (m_meas * m_t >= 0)
#    n_ok = (n_meas >= 0.03 * abs(n_t)) and (n_meas <= 1.5 * abs(n_t)) and (n_meas * n_t >= 0)
    m_ok = (m_meas >= 0.45 * abs(m_t)) and (m_meas <= 1.45 * abs(m_t)) and (m_meas * m_t >= 0)
    n_ok = (n_meas >= 0.15 * abs(n_t)) and (n_meas <= 1.45 * abs(n_t)) and (n_meas * n_t >= 0)



#    m_ok = (abs(m_meas - m_t) <= max(0.6, 0.35 * abs(m_t))) and (m_meas * m_t >= 0)
#    n_ok = (abs(n_meas - n_t) <= max(0.6, 0.35 * abs(n_t))) and (n_meas * n_t >= 0)
    preserved = bool(m_ok and n_ok)

#    preserved = (abs(round(m_meas) - int(m_target)) == 0) and (abs(round(n_meas) - int(n_target)) == 0)
    return {
        "m_measured": m_meas,
        "n_measured": n_meas,
        "m_int": int(round(m_meas)),
        "n_int": int(round(n_meas)),
        "topology_preserved": bool(preserved),
    }


def analyze_relaxed_vortex_3d(psi, grid, medium, m_tor, n_pol):
    """
    Постпроцессинг релаксированного 3D-поля psi(x, y, z):
    интегрирование массы по объёму dV, извлечение R0, sigma, n_eff, g-фактора и топологии.
    """
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    dV = grid["dV"]

    energies, _ = evaluate_energy_and_gradient(psi, grid, medium)
    u, n_eff, _ = compute_medium_fields(psi, medium)

    total_norm = float(np.sum(u) * dV)
    if total_norm <= 1e-20:
        return {
            "mass_mev": 0.0,
            "energies": energies,
            "R0_fm": 0.0,
            "sigma_fm": 0.0,
            "aspect": 0.0,
            "n_eff_max": 1.0,
            "n_eff_mean": 1.0,
            "g_factor": 2.0,
            "chi_max": 0.0,
            "buckling_ok": False,
            "particle_survived": False,

            "topology": {
                "m_measured": 0.0,
                "n_measured": 0.0,
                "m_int": 0,
                "n_int": 0,
                "topology_preserved": False,
            },
        }

    # Взвешенные по 3D-плотности поля u = |psi|^2 геометрические размеры тора
    rho = np.sqrt(X**2 + Y**2)
    R0_fm = float(np.sum(rho * u) * dV / total_norm)

    dist_sq = (rho - R0_fm) ** 2 + Z**2
    sigma_fm = float(np.sqrt(np.sum(dist_sq * u) * dV / total_norm))
    aspect = sigma_fm / max(R0_fm, 1e-6)

    # Показатель преломления (пиковый и средневзвешенный по керну)
    n_eff_max = float(np.max(n_eff))
    n_eff_mean = float(np.sum(n_eff * u) * dV / total_norm)

    # Гиромагнитное отношение (топологическая двойка Хопфа + форм-фактор тора и среды)
    g_factor = 2.0 * (1.0 + (ALPHA_FS / (2.0 * np.pi)) * n_eff_mean * (1.0 + 0.25 * (aspect ** 2)))

    # Проверка сохранения топологии (Тест №2)
    topo = measure_winding_numbers_3d(psi, grid, R0_fm, m_tor, n_pol)

    # Полная масса солитона из 3D-интеграла энергии (МэВ)
    mass_mev = float(abs(energies["E_total"]))
    chi_max = float(energies.get("chi_max", 0.0))
    buckling_ok = bool(energies.get("buckling_ok", True))
    particle_survived = bool(buckling_ok and topo["topology_preserved"] and (mass_mev > 0.0))

    return {
        "mass_mev": mass_mev,
        "energies": energies,
        "R0_fm": R0_fm,
        "sigma_fm": sigma_fm,
        "aspect": aspect,
        "n_eff_max": n_eff_max,
        "n_eff_mean": n_eff_mean,
        "g_factor": float(g_factor),
        "chi_max": chi_max,
        "buckling_ok": buckling_ok,
        "particle_survived": particle_survived,
        "topology": topo,
    }

