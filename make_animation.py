import numpy as np
import plotly.graph_objects as go
from ribbon_renderer import draw_bh_ribbons_3d, find_vortex_centers_3d


def build_vortex_time_animation(psi, grid, azim_deg=45.0, elev_deg=25.0, zoom=1.6, phase_t=0.0, time_label=""):
    # Эволюция волны во времени psi(r, t) = psi(r) * exp(-i * phase_t)
    # Визуализируем огибающую + бегущую фазовую волну Re(psi * exp(-i * phase_t)) при неподвижной камере
    psi_t = psi * np.exp(-1j * float(phase_t))
    amp = np.abs(psi_t)
    wave_field = (amp ** 2) * (0.65 + 0.35 * np.cos(np.angle(psi_t)))
    u_max = float(np.max(wave_field))
    if u_max <= 0:
        u_max = 1.0

    X_flat = grid["X"].flatten()
    Y_flat = grid["Y"].flatten()
    Z_flat = grid["Z"].flatten()
    V_flat = wave_field.flatten()

    # Перевод углов камеры и зума в координаты Plotly scene.camera.eye
    r_cam = 2.6 / max(float(zoom), 0.2)
    az0 = np.radians(float(azim_deg))
    el0 = np.radians(float(elev_deg))

    eye_x0 = r_cam * np.cos(el0) * np.cos(az0)
    eye_y0 = r_cam * np.cos(el0) * np.sin(az0)
    eye_z0 = r_cam * np.sin(el0)

    # Базовый кадр (фиксированный ракурс и приближение)
    base_trace = go.Isosurface(
        x=X_flat,
        y=Y_flat,
        z=Z_flat,
        value=V_flat,
        isomin=0.25 * u_max,
        isomax=0.85 * u_max,
        surface_count=3,
        colorscale="Turbo",
        opacity=0.70,
        caps=dict(x_show=False, y_show=False, z_show=False),
    )

    # Генерация кадров плавного облёта вокруг зафиксированного ракурса и зума
    frames = []
    angles = [0.0]
#    angles = np.linspace(0.0, 2.0 * np.pi, int(n_frames), endpoint=False)
    

    for idx, d_az in enumerate(angles):
        az = az0 + d_az
        ex = r_cam * np.cos(el0) * np.cos(az)
        ey = r_cam * np.cos(el0) * np.sin(az)
        ez = eye_z0
        frames.append(
            go.Frame(
                name=f"f_{idx}",
                layout=dict(
                    scene=dict(
                        camera=dict(
                            eye=dict(x=float(ex), y=float(ey), z=float(ez)),
                            up=dict(x=0.0, y=0.0, z=1.0),
                        )
                    )
                ),
            )
        )

    fig = go.Figure(data=[base_trace], frames=frames)
    fig.update_layout(
 #       title=f"3D-анимация узла (ракурс: {azim_deg:.0f}°, наклон: {elev_deg:.0f}°, зум: {zoom:.1f}×)",

        title=f"3D-анимация узла | {time_label} (ракурс: {azim_deg:.0f}°, наклон: {elev_deg:.0f}°, зум: {zoom:.1f}×)",

        height=560,
        margin=dict(l=0, r=0, t=45, b=0),
        scene=dict(
            aspectmode="data",
            camera=dict(
                eye=dict(x=float(eye_x0), y=float(eye_y0), z=float(eye_z0)),
                up=dict(x=0.0, y=0.0, z=1.0),
            ),
        ),
    )
    return fig

def build_snapshot_history_pngs(psi, grid, azim_deg=45.0, elev_deg=25.0, zoom=1.6, n_frames=36):
    """Генерирует PNG-кадры 3D-лент (ширина B, толщина H, цвет u) с тройным крестом центра."""
    import io
    import matplotlib.pyplot as plt

    L_lim = float(grid["L_half"]) / max(float(zoom), 0.3)
    init_centers = find_vortex_centers_3d(psi, grid)
    png_list = []

    for k in range(int(n_frames)):
        phase_t = 2.0 * np.pi * (k / float(n_frames))
        fig = plt.figure(figsize=(5.5, 5.0), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        draw_bh_ribbons_3d(ax, psi, grid, phase_shift=phase_t, init_centers=init_centers)
        ax.set_xlim(-L_lim, L_lim)
        ax.set_ylim(-L_lim, L_lim)
        ax.set_zlim(-L_lim, L_lim)
        ax.view_init(elev=float(elev_deg), azim=float(azim_deg))
        ax.set_title(f"Snapshot {k + 1}/{n_frames} | фаза {k}/{n_frames} T", fontsize=10)
        ax.set_xlabel("X (fm)", fontsize=8)
        ax.set_ylabel("Y (fm)", fontsize=8)
        ax.set_zlabel("Z (fm)", fontsize=8)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        png_list.append(buf.getvalue())

    return png_list


def build_vortex_evolution_gif(psi_frames, grid, azim_deg=45.0, elev_deg=25.0, zoom=1.6, fps=7):
    """Собирает GIF 3D-лент (B x H, цвет u) по шагам эволюции с крестами центров до и после распада."""
    import io
    import matplotlib.pyplot as plt
    from PIL import Image

    L_lim = float(grid["L_half"]) / max(float(zoom), 0.3)
    init_centers = find_vortex_centers_3d(psi_frames[0], grid)
    pil_frames = []
    total = len(psi_frames)

    for k, psi_k in enumerate(psi_frames):
        fig = plt.figure(figsize=(5.5, 5.0), dpi=110)
        ax = fig.add_subplot(111, projection="3d")
        draw_bh_ribbons_3d(ax, psi_k, grid, phase_shift=0.3 * k, init_centers=init_centers)
        ax.set_xlim(-L_lim, L_lim)
        ax.set_ylim(-L_lim, L_lim)
        ax.set_zlim(-L_lim, L_lim)
        ax.view_init(elev=float(elev_deg), azim=float(azim_deg))
        ax.set_title(f"Шаг эволюции {k + 1}/{total}", fontsize=10)
        ax.set_xlabel("X (fm)", fontsize=8)
        ax.set_ylabel("Y (fm)", fontsize=8)
        ax.set_zlabel("Z (fm)", fontsize=8)
        plt.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        pil_frames.append(Image.open(buf).convert("P", palette=Image.ADAPTIVE))

    gif_buf = io.BytesIO()
    first = pil_frames.pop(0)
    first.save(gif_buf, format="GIF", save_all=True, append_images=pil_frames,
               duration=int(1000 / max(int(fps), 1)), loop=0)
    gif_buf.seek(0)
    return gif_buf.getvalue()


def build_vortex_gif(psi, grid, azim_deg=45.0, elev_deg=25.0, zoom=1.6, n_frames=16, fps=8):
    """Собирает зацикленную GIF-анимацию 3D-лент B x H."""
    import io
    from PIL import Image

    png_list = build_snapshot_history_pngs(
        psi, grid, azim_deg=azim_deg, elev_deg=elev_deg, zoom=zoom, n_frames=n_frames
    )
    frames = [Image.open(io.BytesIO(b)).convert("P", palette=Image.ADAPTIVE) for b in png_list]
    gif_buf = io.BytesIO()
    first_frame = frames.pop(0)
    first_frame.save(
        gif_buf, format="GIF", save_all=True, append_images=frames,
        duration=int(1000 / max(int(fps), 1)), loop=0,
    )
    gif_buf.seek(0)
    return gif_buf.getvalue()

def render_six_particles_gifs(benchmark, lifetimes_s, k_life, r_e_fm, n_grid, max_steps, medium, cross_media, is_radiative, n_frames, azim_deg, elev_deg, zoom_val, prog_bar):
    from preprocessor import build_3d_grid, init_hopf_vortex_3d
    from solution import relax_vortex_3d

    tau_min = min(v for v in lifetimes_s.values() if np.isfinite(v))
    gifs_6 = []
    for idx, (p_name, (m_p, n_p), _, _) in enumerate(benchmark):
        tau_p = float(lifetimes_s.get(p_name, 1e-20))
        if np.isfinite(tau_p):
            steps_p = int(np.clip(180 * ((k_life * tau_p / tau_min) ** 0.06), 180, 900))
            tau_lbl = f"T={k_life:.0f}τ ({k_life * tau_p:.2e} с)"
        else:
            steps_p = int(np.clip(360 * (k_life / 5.0), 240, 720))
            tau_lbl = f"T={k_life:.0f}τ_ref (стабилен)"
        prog_bar.progress(idx / 6.0, text=f"Расчёт и рендер GIF ({idx + 1}/6): {p_name} ({m_p}, {n_p}) | {tau_lbl}...")
