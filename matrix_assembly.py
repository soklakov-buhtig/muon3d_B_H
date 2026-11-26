import numpy as np
from constants import HBAR_C_MEV_FM, R_CORE_VAC_FM, L_VAC_FM


def _grad_3d(psi, dx):
    """Центральные конечные разности первого порядка в 3D (open BC: ноль на границе)."""
    gx = np.zeros_like(psi)
    gy = np.zeros_like(psi)
    gz = np.zeros_like(psi)
    inv_2dx = 0.5 / dx
    gx[1:-1, :, :] = (psi[2:, :, :] - psi[:-2, :, :]) * inv_2dx
    gy[:, 1:-1, :] = (psi[:, 2:, :] - psi[:, :-2, :]) * inv_2dx
    gz[:, :, 1:-1] = (psi[:, :, 2:] - psi[:, :, :-2]) * inv_2dx
    return gx, gy, gz


def _div_3d(fx, fy, fz, dx):
    """Дивергенция векторного поля (fx, fy, fz) в 3D."""
    div = np.zeros_like(fx)
    inv_2dx = 0.5 / dx
    div[1:-1, :, :] += (fx[2:, :, :] - fx[:-2, :, :]) * inv_2dx
    div[:, 1:-1, :] += (fy[:, 2:, :] - fy[:, :-2, :]) * inv_2dx
    div[:, :, 1:-1] += (fz[:, :, 2:] - fz[:, :, :-2]) * inv_2dx
    return div


def compute_medium_fields(psi, medium):
    """Вычисляет локальную плотность u = |psi|^2, показатель преломления n_eff и его производную dn/du."""
    u = np.real(psi * np.conj(psi))
    alpha = float(medium["alpha_nl"])
    gamma = float(medium["gamma_nl"])

    denom = 1.0 + gamma * u
    n_eff = 1.0 + (alpha * u) / denom
    dn_du = alpha / (denom ** 2)
    return u, n_eff, dn_du


def evaluate_energy_and_gradient(psi, grid, medium, use_mask=True):
    """
    Вычисляет полную 3D-энергию E[psi] (E1 + E2 + E3 + E4 в МэВ)
    и точную вариационную производную dE / d(psi*) для шага leapfrog.
    """
    dx = grid["dx"]
    dV = grid["dV"]
    mask = grid["boundary_mask"]

    alpha = float(medium["alpha_nl"])
    gamma = float(medium["gamma_nl"])
    beta = float(medium["beta_nl"])
    eta = float(medium["eta_visc"])
    delta = float(medium["delta_pack"])

    u, n_eff, dn_du = compute_medium_fields(psi, medium)
    gx, gy, gz = _grad_3d(psi, dx)
    grad_sq = np.real(gx * np.conj(gx) + gy * np.conj(gy) + gz * np.conj(gz))

    # Масштабная переводная константа плотности энергии (МэВ / фм^3)
    e_scale = HBAR_C_MEV_FM / (L_VAC_FM ** 2)

    # === E1: Волноводное самозахватывание с насыщением (притягивающий потенциал) ===
    # Интеграл от (n_eff(u) - 1) du = (alpha/gamma) * (u - ln(1 + gamma*u)/gamma)
    if gamma > 1e-12:
        pot_e1 = (alpha / gamma) * (u - np.log1p(gamma * u) / gamma)
    else:
        pot_e1 = 0.5 * alpha * (u ** 2)
    dens_e1 = -e_scale * pot_e1
    grad_e1 = -e_scale * (n_eff - 1.0) * psi

    # === E2: Кинетическая градиентная энергия с замедлением волны c_eff = c / n_eff ===
    kin_coeff = (0.5 / np.pi) * HBAR_C_MEV_FM / n_eff
    dens_e2 = kin_coeff * grad_sq
    # Вариационная производная: -div(kin_coeff * grad(psi)) + (d(kin_coeff)/du) * |grad(psi)|^2 * psi
    div_term = _div_3d(kin_coeff * gx, kin_coeff * gy, kin_coeff * gz, dx)
    dkin_du = -kin_coeff * (dn_du / n_eff)
    grad_e2 = -div_term + dkin_du * grad_sq * psi

    # === E3: Кручение и сдвиговое прощёлкивание (через плотность тока J = Im(psi* grad psi)) ===
    jx = np.imag(np.conj(psi) * gx)
 #   alias_scale = float(grid.get("alias_scale", 1.0))
    alias_scale = float(grid.get("alias_scale", 1.0)) if use_mask else 1.0

    jx = np.imag(np.conj(psi) * gx) * alias_scale
    jy = np.imag(np.conj(psi) * gy) * alias_scale
    jz = np.imag(np.conj(psi) * gz) * alias_scale
    j_sq = jx**2 + jy**2 + jz**2


#    chi_local = eta * np.sqrt(j_sq + 1e-16) * (R_CORE_VAC_FM ** 2)
    chi_local = eta * np.sqrt(j_sq + 1e-16) * (R_CORE_VAC_FM ** 2) / (1.0 + delta * u)

    buckling_rel = 1.0 + (chi_local ** 3) / (1.0 + delta * (chi_local ** 3))
    torsion_scale = beta * HBAR_C_MEV_FM * (R_CORE_VAC_FM ** 2)
    chi_max = float(np.max(chi_local))
    buckling_ok = bool(chi_max < 1.0)

    dens_e3 = torsion_scale * j_sq / buckling_rel

    # Вариационный вклад кручения через ток: -i * div(2 * torsion_eff * J * psi)
    torsion_eff = torsion_scale / buckling_rel
    grad_e3 = -1j * _div_3d(2.0 * torsion_eff * jx * psi,
                            2.0 * torsion_eff * jy * psi,
                            2.0 * torsion_eff * jz * psi, dx)

    # === E4: Упаковка и жёсткий керн (предотвращение коллапса в сингулярность) ===
    pack_coeff = delta * e_scale
    dens_e4 = 0.5 * pack_coeff * (u ** 3)
    grad_e4 = 1.5 * pack_coeff * (u ** 2) * psi

    # Суммирование энергий по 3D-объёму (МэВ)
    E1 = float(np.sum(dens_e1) * dV)
    E2 = float(np.sum(dens_e2) * dV)
    E3 = float(np.sum(dens_e3) * dV)
    E4 = float(np.sum(dens_e4) * dV)
    E_total = E1 + E2 + E3 + E4

    # Полный вариационный градиент dE / d(psi*) с учётом граничной маски open BC
#    dE_dpsi = (grad_e1 + grad_e2 + grad_e3 + grad_e4) * mask
    dE_dpsi = (grad_e1 + grad_e2 + grad_e3 + grad_e4) * (mask if use_mask else 1.0)


    energies = {
        "E_total": E_total,
        "E1": E1,
        "E2": E2,
        "E3": E3,
        "E4": E4,
        "chi_max": chi_max,
        "buckling_ok": buckling_ok,
    }

    return energies, dE_dpsi
