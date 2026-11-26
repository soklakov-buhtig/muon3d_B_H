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

