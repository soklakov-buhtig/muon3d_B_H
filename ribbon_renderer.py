import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter, label


def compute_bh_fields(psi, grid):
    """Вычисляет плотность энергии u, магнитную индукцию B и напряженность H = B / mu_r."""
    dx = float(grid["dx"])
    u = np.real(psi * np.conj(psi))
    u_max = max(float(np.max(u)), 1e-12)
    u_norm = u / u_max

    bx = np.zeros_like(u)
    by = np.zeros_like(u)
    bz = np.zeros_like(u)
    bx[1:-1, :, :] = np.imag(np.conj(psi[1:-1, :, :]) * (psi[2:, :, :] - psi[:-2, :, :])) * (0.5 / dx)
    by[:, 1:-1, :] = np.imag(np.conj(psi[:, 1:-1, :]) * (psi[:, 2:, :] - psi[:, :-2, :])) * (0.5 / dx)
    bz[:, :, 1:-1] = np.imag(np.conj(psi[:, :, 1:-1]) * (psi[:, :, 2:] - psi[:, :, :-2])) * (0.5 / dx)

    b_mag = np.sqrt(bx**2 + by**2 + bz**2)
    b_max = max(float(np.max(b_mag)), 1e-12)
    b_norm = b_mag / b_max

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
    """Находит 1 общий центр масс, пока вихрь цельный, или центры осколков после распада."""
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    u = np.real(psi * np.conj(psi))
    u_max = float(np.max(u))
    if u_max <= 1e-16:
        return [np.zeros(3)]

    mask = u >= float(rel_thresh) * u_max
    if not np.any(mask):
        return [np.zeros(3)]

    labeled, n_comp = label(mask, structure=np.ones((3, 3, 3), dtype=int))
    comp_masses = []
    tot_mask_u = float(np.sum(u[mask]))
    for c_id in range(1, n_comp + 1):
        c_mask = (labeled == c_id)
        m_val = float(np.sum(u[c_mask]))
        if m_val > 0.02 * tot_mask_u:
            cx = float(np.sum(X[c_mask] * u[c_mask]) / m_val)
            cy = float(np.sum(Y[c_mask] * u[c_mask]) / m_val)
            cz = float(np.sum(Z[c_mask] * u[c_mask]) / m_val)
            comp_masses.append((m_val, np.array([cx, cy, cz])))

    if not comp_masses:
        return [np.zeros(3)]
    comp_masses.sort(key=lambda item: item[0], reverse=True)
    return [c for _, c in comp_masses[:max_centers]]
def draw_triple_cross_3d(ax, center, arm_len, color="black", lw=1.6, ls="-", alpha=0.95):
    """Рисует тройной ортогональный 3D-крест (оси X, Y, Z) в указанной точке."""
    cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
    L = float(arm_len)
    ax.plot([cx - L, cx + L], [cy, cy], [cz, cz], color=color, lw=lw, ls=ls, alpha=alpha, zorder=20)
    ax.plot([cx, cx], [cy - L, cy + L], [cz, cz], color=color, lw=lw, ls=ls, alpha=alpha, zorder=20)
    ax.plot([cx, cx], [cy, cy], [cz - L, cz + L], color=color, lw=lw, ls=ls, alpha=alpha, zorder=20)
def draw_bh_ribbons_3d(ax, psi, grid, phase_shift=0.0, init_centers=None, n_ribbons=10, n_steps=85):
    """Рисует гладкие 3D-ленты (ширина ~ B, толщина ~ H, цвет ~ u) и тройные 3D-кресты."""
    X, Y, Z = grid["X"], grid["Y"], grid["Z"]
    L_half, dx = float(grid["L_half"]), float(grid["dx"])
    u, u_max, bx, by, bz, b_norm, h_norm = compute_bh_fields(psi, grid)

    bx_s = gaussian_filter(bx, sigma=0.95)
    by_s = gaussian_filter(by, sigma=0.95)
    bz_s = gaussian_filter(bz, sigma=0.95)

    core_idx = np.argwhere(u >= 0.45 * u_max)
    if len(core_idx) == 0:
        core_idx = np.argwhere(u >= 0.15 * u_max)

    if len(core_idx) > 0:
        stride = max(1, len(core_idx) // max(int(n_ribbons), 1))
        seeds_idx = core_idx[::stride][:int(n_ribbons)]
        seeds = np.array([[X[i, j, k], Y[i, j, k], Z[i, j, k]] for i, j, k in seeds_idx], dtype=float)

        ds = 0.35 * dx
        traj = np.zeros((len(seeds), n_steps, 3), dtype=float)
        traj[:, 0, :] = seeds
        for s in range(1, n_steps):
            p = traj[:, s - 1, :]
            v1 = np.column_stack([_interp3d(bx_s, grid, p), _interp3d(by_s, grid, p), _interp3d(bz_s, grid, p)])
            v1 /= np.maximum(np.linalg.norm(v1, axis=1, keepdims=True), 1e-12)
            p_mid = np.clip(p + 0.5 * ds * v1, -0.95 * L_half, 0.95 * L_half)
            v2 = np.column_stack([_interp3d(bx_s, grid, p_mid), _interp3d(by_s, grid, p_mid), _interp3d(bz_s, grid, p_mid)])
            v2 /= np.maximum(np.linalg.norm(v2, axis=1, keepdims=True), 1e-12)
            traj[:, s, :] = np.clip(p + ds * v2, -0.95 * L_half, 0.95 * L_half)
        cmap = plt.get_cmap("turbo")
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
            twist = 0.25 * np.arange(len(pts)) + float(phase_shift)
            e1 = np.cos(twist)[:, None] * n_vec + np.sin(twist)[:, None] * b_vec
            e2 = -np.sin(twist)[:, None] * n_vec + np.cos(twist)[:, None] * b_vec

            w_half = (0.35 + 1.15 * b_line)[:, None] * dx
            t_half = (0.18 + 0.75 * h_line)[:, None] * dx

            p_tl = pts - w_half * e1 + t_half * e2
            p_tr = pts + w_half * e1 + t_half * e2
            p_br = pts + w_half * e1 - t_half * e2
            p_bl = pts - w_half * e1 - t_half * e2

            for s in range(len(pts) - 1):
                if u_line[s] < 0.04 * u_max:
                    continue
                col = cmap(norm_c(u_line[s]))
                polys.extend([
                    [p_tl[s], p_tr[s], p_tr[s + 1], p_tl[s + 1]],
                    [p_br[s], p_bl[s], p_bl[s + 1], p_br[s + 1]],
                    [p_tr[s], p_br[s], p_br[s + 1], p_tr[s + 1]],
                    [p_bl[s], p_tl[s], p_tl[s + 1], p_bl[s + 1]],
                ])
                face_colors.extend([col, col, col, col])
        if polys:
            coll = Poly3DCollection(polys, facecolors=face_colors, edgecolors="none", alpha=0.88)
            ax.add_collection3d(coll)

    arm = 0.14 * L_half
    if init_centers is not None:
        for c0 in init_centers:
            draw_triple_cross_3d(ax, c0, arm * 1.15, color="#555555", lw=1.3, ls="--", alpha=0.75)
    cur_centers = find_vortex_centers_3d(psi, grid)
    for idx_c, cc in enumerate(cur_centers):
        c_col = "#111111" if idx_c == 0 else "#d62728"
        draw_triple_cross_3d(ax, cc, arm, color=c_col, lw=1.8, ls="-", alpha=0.95)
