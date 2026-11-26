import matplotlib.pyplot as plt
from matplotlib import cm, colors

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

def draw_bh_ribbons_3d(ax, psi, grid, phase_shift=0.0, init_centers=None, n_ribbons=12, n_steps=38):
    """
    Рисует 3D-ленты (ширина ~ B, толщина ~ H = B/mu_r, цвет ~ плотность энергии u)
    и тройные 3D-кресты центров вихря до и после распада.
    """
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    L_half = float(grid["L_half"])
    dx = float(grid["dx"])
    u, u_max, bx, by, bz, b_norm, h_norm = compute_bh_fields(psi, grid)

    # Выбор стартовых точек в плотном ядре (u >= 0.45 * u_max)
    core_idx = np.argwhere(u >= 0.45 * u_max)
    if len(core_idx) == 0:
        core_idx = np.argwhere(u >= 0.15 * u_max)

    if len(core_idx) > 0:
        stride = max(1, len(core_idx) // max(int(n_ribbons), 1))
        seeds_idx = core_idx[::stride][:int(n_ribbons)]
        seeds = np.array([[X[i, j, k], Y[i, j, k], Z[i, j, k]] for i, j, k in seeds_idx], dtype=float)

        # Трассировка силовых линий методом РК-2 вдоль поля B
        ds = 0.65 * dx
        traj = np.zeros((len(seeds), n_steps, 3), dtype=float)
        traj[:, 0, :] = seeds
        for s in range(1, n_steps):
            p = traj[:, s - 1, :]
            v1 = np.column_stack([_interp3d(bx, grid, p), _interp3d(by, grid, p), _interp3d(bz, grid, p)])
            n1 = np.linalg.norm(v1, axis=1, keepdims=True)
            v1 = np.where(n1 > 1e-12, v1 / np.maximum(n1, 1e-12), 0.0)
            p_mid = np.clip(p + 0.5 * ds * v1, -0.95 * L_half, 0.95 * L_half)
            v2 = np.column_stack([_interp3d(bx, grid, p_mid), _interp3d(by, grid, p_mid), _interp3d(bz, grid, p_mid)])
            n2 = np.linalg.norm(v2, axis=1, keepdims=True)
            v2 = np.where(n2 > 1e-12, v2 / np.maximum(n2, 1e-12), 0.0)
            traj[:, s, :] = np.clip(p + ds * v2, -0.95 * L_half, 0.95 * L_half)

        cmap = cm.get_cmap("turbo")

 #       cmap = cm.get_cmap("turbo")
        norm_c = colors.Normalize(vmin=0.05 * u_max, vmax=u_max)
        polys, face_colors = [], []

        for r_i in range(len(seeds)):
            pts = traj[r_i]
            u_line = _interp3d(u, grid, pts)
            b_line = _interp3d(b_norm, grid, pts)
            h_line = _interp3d(h_norm, grid, pts)

            tang = np.gradient(pts, axis=0)
            tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-12)
            ref = np.tile(np.array([0.0, 0.0, 1.0]), (len(pts), 1))
            ref[np.abs(tang[:, 2]) > 0.85] = np.array([1.0, 0.0, 0.0])

            n_vec = np.cross(tang, ref)
            n_vec /= np.maximum(np.linalg.norm(n_vec, axis=1, keepdims=True), 1e-12)
            b_vec = np.cross(tang, n_vec)

            # Перекрут сечения (ширина ~ B по n_vec, толщина ~ H по b_vec)
            twist = 0.25 * np.arange(len(pts)) + float(phase_shift)
            e1 = np.cos(twist)[:, None] * n_vec + np.sin(twist)[:, None] * b_vec
            e2 = -np.sin(twist)[:, None] * n_vec + np.cos(twist)[:, None] * b_vec

            w_half = (0.35 + 1.15 * b_line)[:, None] * dx
            t_half = (0.18 + 0.75 * h_line)[:, None] * dx

            p_top_l = pts - w_half * e1 + t_half * e2
            p_top_r = pts + w_half * e1 + t_half * e2
            p_bot_r = pts + w_half * e1 - t_half * e2
            p_bot_l = pts - w_half * e1 - t_half * e2

            for s in range(len(pts) - 1):
                if u_line[s] < 0.04 * u_max:
                    continue
                col = cmap(norm_c(u_line[s]))
                # Верхняя, нижняя и две боковые грани объемной ленты B x H
                polys.append([p_top_l[s], p_top_r[s], p_top_r[s + 1], p_top_l[s + 1]])
                polys.append([p_bot_r[s], p_bot_l[s], p_bot_l[s + 1], p_bot_r[s + 1]])
                polys.append([p_top_r[s], p_bot_r[s], p_bot_r[s + 1], p_top_r[s + 1]])
                polys.append([p_bot_l[s], p_top_l[s], p_top_l[s + 1], p_bot_l[s + 1]])
                face_colors.extend([col, col, col, col])

        if polys:
            coll = Poly3DCollection(polys, facecolors=face_colors, edgecolors="none", alpha=0.88)
            ax.add_collection3d(coll)

    # Тройные 3D-кресты: серый пунктирный до распада, яркий чёрный/красный для текущих центров
    arm = 0.14 * L_half
    if init_centers is not None:
        for c0 in init_centers:
            draw_triple_cross_3d(ax, c0, arm * 1.15, color="#555555", lw=1.3, ls="--", alpha=0.75)
    cur_centers = find_vortex_centers_3d(psi, grid)
    for idx_c, cc in enumerate(cur_centers):
        c_col = "#111111" if idx_c == 0 else "#d62728"
        draw_triple_cross_3d(ax, cc, arm, color=c_col, lw=1.8, ls="-", alpha=0.95)
