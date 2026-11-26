import numpy as np
from matplotlib import cm, colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import maximum_filter


def compute_bh_fields(psi, grid):
    """Вычисляет плотность энергии u, магнитную индукцию B и напряженность H = B / mu_r."""
    dx = float(grid["dx"])
    u = np.real(psi * np.conj(psi))
    u_max = max(float(np.max(u)), 1e-12)
    u_norm = u / u_max

    # Вектор индукции B пропорционален вихревому потоку Im(psi* grad psi)
    bx = np.zeros_like(u)
    by = np.zeros_like(u)
    bz = np.zeros_like(u)
    bx[1:-1, :, :] = np.imag(np.conj(psi[1:-1, :, :]) * (psi[2:, :, :] - psi[:-2, :, :])) * (0.5 / dx)
    by[:, 1:-1, :] = np.imag(np.conj(psi[:, 1:-1, :]) * (psi[:, 2:, :] - psi[:, :-2, :])) * (0.5 / dx)
    bz[:, :, 1:-1] = np.imag(np.conj(psi[:, :, 1:-1]) * (psi[:, :, 2:] - psi[:, :, :-2])) * (0.5 / dx)

    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    b_max = max(float(np.max(b_mag)), 1e-12)
    b_norm = b_mag / b_max

    # Нелинейная проницаемость Борна-Инфельда mu_r = sqrt(1 - u_norm) -> H = B / mu_r
    mu_r = np.sqrt(np.maximum(0.08, 1.0 - 0.92 * u_norm))
    h_norm = np.minimum(2.5, b_norm / mu_r)
    return u, u_max, bx, by, bz, b_norm, h_norm


def _interp3d(field, grid, pts):
    """Трилинейная интерполяция скалярного 3D-поля в точках pts (M, 3)."""
    L = float(grid["L_half"])
    dx = float(grid["dx"])
    N = field.shape[0]
    u = np.clip((pts[:, 0] + L) / dx, 0.0, N - 1.0001)
    v = np.clip((pts[:, 1] + L) / dx, 0.0, N - 1.0001)
    w = np.clip((pts[:, 2] + L) / dx, 0.0, N - 1.0001)
    i0, j0, k0 = u.astype(int), v.astype(int), w.astype(int)
    du, dv, dw = u - i0, v - j0, w - k0
    c00 = field[i0, j0, k0] * (1 - du) + field[i0 + 1, j0, k0] * du
    c01 = field[i0, j0, k0 + 1] * (1 - du) + field[i0 + 1, j0, k0 + 1] * du
    c10 = field[i0, j0 + 1, k0] * (1 - du) + field[i0 + 1, j0 + 1, k0] * du
    c11 = field[i0, j0 + 1, k0 + 1] * (1 - du) + field[i0 + 1, j0 + 1, k0 + 1] * du
    return (c00 * (1 - dv) + c10 * dv) * (1 - dw) + (c01 * (1 - dv) + c11 * dv) * dw

def find_vortex_centers_3d(psi, grid, rel_thresh=0.35, max_centers=6):
    """Находит центр(ы) вихревых сгустков по локальным максимумам плотности энергии u."""
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    u = np.real(psi * np.conj(psi))
    u_max = float(np.max(u))
    if u_max <= 1e-16:
        return [np.zeros(3)]

    # Если вихрь в целом — проверяем глобальный центр масс плотного ядра
    core_mask = u >= 0.50 * u_max
    w_sum = float(np.sum(u[core_mask]))
    com = np.array([
        float(np.sum(X[core_mask] * u[core_mask]) / w_sum),
        float(np.sum(Y[core_mask] * u[core_mask]) / w_sum),
        float(np.sum(Z[core_mask] * u[core_mask]) / w_sum),
    ])

    # Поиск локальных пиков сгустков (если вихрь разделился на фрагменты при распаде)
    local_max = (u == maximum_filter(u, size=5)) & (u >= rel_thresh * u_max)
    idx = np.argwhere(local_max)
    if len(idx) <= 1:
        return [com]

    vals = [float(u[i, j, k]) for i, j, k in idx]
    order = np.argsort(vals)[::-1]
    min_dist = 3.5 * float(grid["dx"])
    centers = []
    for o in order:
        i, j, k = idx[o]
        pt = np.array([float(X[i, j, k]), float(Y[i, j, k]), float(Z[i, j, k])])
        if all(np.linalg.norm(pt - c) >= min_dist for c in centers):
            centers.append(pt)
        if len(centers) >= max_centers:
            break
    return centers if centers else [com]


def draw_triple_cross_3d(ax, center, arm_len, color="black", lw=1.6, ls="-", alpha=0.95):
    """Рисует тройной ортогональный 3D-крест (оси X, Y, Z) в указанной точке."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    L = float(arm_len)
    ax.plot([cx - L, cx + L], [cy, cy], [cz, cz], color=color, lw=lw, ls=ls, alpha=alpha, zorder=20)
    ax.plot([cx, cx], [cy - L, cy + L], [cz, cz], color=color, lw=lw, ls=ls, alpha=alpha, zorder=20)
    ax.plot([cx, cx], [cy, cy], [cz - L, cz + L], color=color, lw=lw, ls=ls, alpha=alpha, zorder=20)
