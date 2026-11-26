import streamlit as st
import numpy as np
import plotly.graph_objects as go


#from calibrate_medium import calibrate_single_and_scan_spectrum
#from calibrate_medium import calibrate_single_and_scan_spectrum, cross_calibrate_all_particles

from calibrate_medium import (
    calibrate_single_and_scan_spectrum,
    cross_calibrate_all_particles,
    FOUND_PARTICLES_BENCHMARK,
)


from constants import DEFAULT_MEDIUM, DEFAULT_GRID, M_E_MEV, M_MU_MEV, R_E_COMPTON_FM, R_MU_COMPTON_FM
from preprocessor import build_3d_grid, init_hopf_vortex_3d, init_noise_field_3d
from solution import relax_vortex_3d
from postprocessor import analyze_relaxed_vortex_3d
#from make_animation import build_vortex_rotation_animation
#from make_animation import build_vortex_time_animation

#from make_animation import build_vortex_time_animation, build_snapshot_history_pngs
#from make_animation import build_vortex_time_animation, build_vortex_gif

from make_animation import (
    build_vortex_time_animation,
    build_vortex_gif,
    build_vortex_evolution_gif,
)



st.set_page_config(page_title="Muon 3D Vortex Solver", layout="wide")
st.title("3D Soliton Relaxation Solver (dt = τ / 2)")

with st.sidebar:
    st.header("1. Топология и начальный масштаб")
    preset = st.selectbox("Пресет частицы", ["Электрон (2, 3)", "Мюон (37, 73)", "Мюон (73, 37)", "Свой узел"])
    if preset == "Электрон (2, 3)":
        def_m, def_n, def_r, def_s = 2, 3, float(R_E_COMPTON_FM), float(R_E_COMPTON_FM * 0.35)
    elif preset == "Мюон (37, 73)":
        def_m, def_n, def_r, def_s = 37, 73, float(R_MU_COMPTON_FM), float(R_MU_COMPTON_FM * 0.35)
    elif preset == "Мюон (73, 37)":
        def_m, def_n, def_r, def_s = 73, 37, float(R_MU_COMPTON_FM), float(R_MU_COMPTON_FM * 0.35)
    else:
        def_m, def_n, def_r, def_s = 2, 3, 100.0, 35.0

    col_m, col_n = st.columns(2)
    m_tor = col_m.number_input("m_tor", min_value=1, max_value=500, value=def_m, step=1)
    n_pol = col_n.number_input("n_pol", min_value=1, max_value=500, value=def_n, step=1)
    R0_init = st.number_input("Начальный R0 (fm)", min_value=0.1, max_value=10000.0, value=def_r, format="%.3f")
    sig_init = st.number_input("Начальная толщина σ (fm)", min_value=0.05, max_value=5000.0, value=def_s, format="%.3f")

    st.header("2. Среда и шаг dt = τ / 2")
    alpha_nl = st.number_input("alpha_nl", value=float(DEFAULT_MEDIUM["alpha_nl"]), format="%.6g")
    gamma_nl = st.number_input("gamma_nl", value=float(DEFAULT_MEDIUM["gamma_nl"]), format="%.6g")
    beta_nl = st.number_input("beta_nl", value=float(DEFAULT_MEDIUM["beta_nl"]), format="%.6g")
    eta_visc = st.number_input("eta_visc", value=float(DEFAULT_MEDIUM["eta_visc"]), format="%.6g")
    delta_pack = st.number_input("delta_pack", value=float(DEFAULT_MEDIUM["delta_pack"]), format="%.6g")
    tau = st.number_input("tau (шаг dt = tau / 2)", min_value=0.01, max_value=10.0, value=float(DEFAULT_MEDIUM["tau"]), format="%.3f")

    st.header("3. 3D-сетка (open BC)")
    N_grid = st.slider("Размер куба N³", min_value=24, max_value=80, value=int(DEFAULT_GRID["N"]), step=8)
    box_scale = st.slider("Размер бокса (× R0)", min_value=2.5, max_value=6.0, value=float(DEFAULT_GRID["box_scale"]), step=0.5)
    max_steps = st.slider("Макс. шагов релаксации", min_value=100, max_value=2000, value=int(DEFAULT_GRID["max_steps"]), step=100)

medium = {
    "alpha_nl": alpha_nl,
    "gamma_nl": gamma_nl,
    "beta_nl": beta_nl,
    "eta_visc": eta_visc,
    "delta_pack": delta_pack,
    "tau": tau,
}

col_btn1, col_btn2 = st.columns(2)
run_vortex = col_btn1.button("Запустить 3D-релаксацию узла", type="primary", use_container_width=True)
run_noise = col_btn2.button("Тест №1: Устойчивость шума (ψ ≈ 0)", use_container_width=True)
#run_calib = st.button("Запустить калибратор среды (поиск электрона и мюона)", use_container_width=True)
run_calib = st.button("Настроить среду по младшему узлу и пройти спектр (buckling + динамика)", use_container_width=True)
if run_calib:

    with st.spinner(f"1) Калибровка среды на одном узле ({m_tor}, {n_pol}) + 2) последовательный проход спектра..."):
        cal_res = calibrate_single_and_scan_spectrum(
            medium,
            base_topo=(m_tor, n_pol),
            N_grid=32,
            dyn_steps=120,
            max_iter=30,
        )
    st.subheader(f"Калибровка завершена | Лог-ошибка младшего узла: {cal_res['loss']:.5f}")
    st.json(cal_res["medium"])

    st.subheader("Последовательный проход по топологиям (проверка динамики и buckling)")
    rows = []
    for item in cal_res["spectrum"]:
        rows.append(
            {
                "Узел (m, n)": f"{item['topo'][0]}, {item['topo'][1]}",
                "Выжил": "ДА" if item["survived"] else "НЕТ",
                "Buckling OK": "ДА" if item["buckling_ok"] else "СРЫВ",
                "Динамика OK": "ДА" if item["dynamically_stable"] else "НЕТ",
                "Топология OK": "ДА" if item["topology_preserved"] else "НЕТ",
                "chi_max": round(item["chi_max"], 4),
                "Масса (МэВ)": round(item["mass_mev"], 5),
                "R0 (fm)": round(item["R0_fm"], 3),
                "g-фактор": round(item["g_factor"], 6),
            }
        )
    st.dataframe(rows, use_container_width=True)
run_cross = st.button("Проверить инвариантность калибровки по каждой найденной частице", use_container_width=True)
if run_cross:
    prog_bar = st.progress(0, text="Подготовка кросс-калибровки по 6 частицам...")
    live_table = st.empty()
    
    partial_rows = []

    def on_particle_done(done_cnt, total_cnt, p_name, row_dict):
        partial_rows.append(row_dict)
        prog_bar.progress(
            done_cnt / total_cnt,
            text=f"Готово {done_cnt} из {total_cnt}: {p_name} (погр. {row_dict['Погр. массы (%)']}%)",
        )
        live_table.dataframe(partial_rows, use_container_width=True)

    cross_rows = cross_calibrate_all_particles(
        medium,
        N_grid=32,
        dyn_steps=75,
        max_iter=45,
        progress_callback=on_particle_done,
    )
    prog_bar.progress(1.0, text="Кросс-калибровка по всем 6 частицам завершена!")
    st.subheader("Тест независимости констант среды от выбора эталонной частицы")
    live_table.dataframe(cross_rows, use_container_width=True)
    st.session_state["cross_media"] = {
        r["Эталонная частица"]: {
            "alpha_nl": r["alpha_nl"],
            "gamma_nl": r["gamma_nl"],
            "beta_nl": r["beta_nl"],
            "eta_visc": r["eta_visc"],
            "delta_pack": r["delta_pack"],
            "tau": tau,
        }
        for r in cross_rows
    }



if run_noise:
    with st.spinner("Запуск Теста №1 (релаксация малого шума вокруг ψ = 0)..."):
        grid = build_3d_grid(R0_init, N=N_grid, box_scale=box_scale)
        psi_noise = init_noise_field_3d(grid, noise_amp=1e-3)
 #       res = relax_vortex_3d(psi_noise, grid, medium, max_steps=min(max_steps, 300), renormalize=False)

        res = relax_vortex_3d(psi_noise, grid, medium, max_steps=min(max_steps, 300), renormalize=False)

       


#        E_start = res["history"]["E_total"]
        E_start = res["history"][0]["E_total"]

        E_end = res["energies"]["E_total"]
        st.subheader("Результаты Теста №1 (устойчивость на малом шуме)")
        st.write(f"**Начальная энергия шума:** `{E_start:.6e} МэВ` → **Конечная:** `{E_end:.6e} МэВ` (шагов: `{res['steps_done']}`)")
        if abs(E_end) <= abs(E_start) * 1.05:
            st.success("Тест №1 пройден: шум затухает, схема leapfrog (dt = tau / 2) устойчива.")
        else:
            st.error("Внимание: энергия шума выросла — проверьте шаг или коэффициенты среды.")

if run_vortex:
    with st.spinner(f"Релаксация 3D-узла ({m_tor}, {n_pol}) на сетке {N_grid}³..."):
        grid = build_3d_grid(R0_init, N=N_grid, box_scale=box_scale)
        psi0 = init_hopf_vortex_3d(grid, R0_init, sig_init, m_tor, n_pol)
        res = relax_vortex_3d(
            psi0, grid, medium,
            max_steps=max_steps,
            norm_interval=1,
            tol_energy=DEFAULT_GRID["tol_energy"],
            renormalize=False,
        )
        obs = analyze_relaxed_vortex_3d(res["psi"], grid, medium, m_tor, n_pol)
        st.session_state["last_3d_run"] = {
            "psi": res["psi"],
            "grid": grid,
            "m_tor": m_tor,
            "n_pol": n_pol,
        }


    st.subheader(f"Результаты 3D-релаксации ({m_tor}, {n_pol}) | dt = {res['dt']:.3f}")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Масса 3D (МэВ)", f"{obs['mass_mev']:.5f}")
    c2.metric("R0 (fm)", f"{obs['R0_fm']:.3f}")
    c3.metric("σ (fm)", f"{obs['sigma_fm']:.3f}")
    c4.metric("n_eff (max / mean)", f"{obs['n_eff_max']:.3f} / {obs['n_eff_mean']:.3f}")
    c5.metric("g-фактор", f"{obs['g_factor']:.6f}")

    # Тест №2: Сохранение топологии
    topo = obs["topology"]
    
    if topo["topology_preserved"]:
        st.success(f"Тест №2 (топология сохранена): измерено ({topo['m_measured']:.2f}, {topo['n_measured']:.2f}) ≈ ({m_tor}, {n_pol})")
    else:
        st.warning(f"Тест №2 (дрейф топологии): измерено ({topo['m_measured']:.2f}, {topo['n_measured']:.2f}) при цели ({m_tor}, {n_pol}) — попробуйте крупнее сетку N³.")
    if obs.get("buckling_ok", True) and res.get("dynamically_stable", True):
        st.success(f"Динамика и buckling пройдены: chi_max = {obs.get('chi_max', 0.0):.4f} < 1.0, amp_ratio = {res.get('amp_ratio', 1.0):.3f}")
    else:
        st.error(f"Потеря устойчивости: buckling_ok={obs.get('buckling_ok')}, chi_max={obs.get('chi_max', 0.0):.4f}, amp_ratio={res.get('amp_ratio', 1.0):.3f}")

    st.write(
        f"**Энергии (МэВ):** `E_total = {obs['energies']['E_total']:.4f}` | "
        f"`E1 = {obs['energies']['E1']:.4f}` | `E2 = {obs['energies']['E2']:.4f}` | "
        f"`E3 = {obs['energies']['E3']:.4f}` | `E4 = {obs['energies']['E4']:.4f}` | "
        f"**Шагов:** `{res['steps_done']}` (сходимость: `{res['converged']}`)"
    )

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        # 3D-изоповерхность плотности |psi|^2
        u_field = np.abs(res["psi"]) ** 2
        u_max = float(np.max(u_field))
        fig3d = go.Figure(data=go.Isosurface(
            x=grid["X"].flatten(),
            y=grid["Y"].flatten(),
            z=grid["Z"].flatten(),
            value=u_field.flatten(),
            isomin=0.25 * u_max,
            isomax=0.85 * u_max,
            surface_count=3,
            colorscale="Turbo",
            opacity=0.65,
            caps=dict(x_show=False, y_show=False, z_show=False),
        ))
        fig3d.update_layout(title="3D-изоповерхность плотности |ψ|²", height=430, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig3d, use_container_width=True)

    with col_v2:
        # График релаксации энергии по шагам
        steps_arr = [h["step"] for h in res["history"]]
        e_tot_arr = [h["E_total"] for h in res["history"]]
        fig_e = go.Figure()
        fig_e.add_trace(go.Scatter(x=steps_arr, y=e_tot_arr, mode="lines+markers", name="E_total (МэВ)"))
        fig_e.update_layout(title="Сходимость полной 3D-энергии E(t)", xaxis_title="Шаг", yaxis_title="МэВ", height=430)
        st.plotly_chart(fig_e, use_container_width=True)
# === Отдельный блок: Фиксация ориентации и 3D-анимация ===
if "last_3d_run" in st.session_state:
    st.divider()



c_az, c_el, c_zm, c_ph, c_btn1, c_btn2, c_btn3 = st.columns([1, 1, 1, 1, 1.2, 1.0, 1.1])
azim_deg = c_az.slider("Азимут (°)", 0, 360, 45, step=5)
elev_deg = c_el.slider("Наклон (°)", -85, 85, 25, step=5)
zoom_val = c_zm.slider("Приближение (×)", 0.5, 4.0, 1.6, step=0.1)
manual_step = c_ph.slider("Кадр времени t (0..24)", 0, 24, 0, step=1)

with c_btn1:
    st.write("")
    st.write("")
    lock_and_animate = st.button("Запустить эволюцию (3D)", width="stretch")

with c_btn2:
    st.write("")
    st.write("")
    make_snapshots = st.button("Snapshot History", width="stretch")

with c_btn3:
    st.write("")
    st.write("")
    make_6_gifs = st.button("GIF для 6 частиц", width="stretch")

anim_slot = st.empty()

if "last_3d_run" in st.session_state:
    run_data = st.session_state["last_3d_run"]
    if lock_and_animate:
        import time
        t_start = time.time()
        for step_t in range(25):
            phase_t = 2.0 * np.pi * (step_t / 24.0)
            elapsed = time.time() - t_start
            t_str = f"t = {elapsed:.2f} с (фаза {step_t}/24 T)"
            fig_step = build_vortex_time_animation(
                run_data["psi"], run_data["grid"],
                azim_deg=azim_deg, elev_deg=elev_deg,
                zoom=zoom_val, phase_t=phase_t, time_label=t_str,
            )
            anim_slot.plotly_chart(fig_step, width="stretch", key=f"anim_t_{step_t}")
            time.sleep(0.06)
    else:
        fig_preview = build_vortex_time_animation(
            run_data["psi"], run_data["grid"],
            azim_deg=azim_deg, elev_deg=elev_deg,
            zoom=zoom_val,
            phase_t=2.0 * np.pi * (manual_step / 24.0),
            time_label="t = 0.00 с (камера зафиксирована)",
        )
        anim_slot.plotly_chart(fig_preview, width="stretch", key="anim_preview")
elif lock_and_animate or make_snapshots:
    st.warning("Сначала запустите 3D-релаксацию узла или нажмите «GIF для 6 частиц».")



n_snap_frames = st.number_input("Количество кадров GIF", min_value=8, max_value=240, value=16, step=8)

if make_snapshots and "last_3d_run" in st.session_state:
    run_data = st.session_state["last_3d_run"]
    with st.spinner("Сборка GIF-анимации..."):
        st.session_state["vortex_gif"] = build_vortex_gif(
            run_data["psi"], run_data["grid"],
            azim_deg=azim_deg, elev_deg=elev_deg,
            zoom=zoom_val, n_frames=int(n_snap_frames),
        )

if "vortex_gif" in st.session_state:
    gif_bytes = st.session_state["vortex_gif"]
    st.image(gif_bytes)
    st.download_button(
        label="Скачать GIF-анимацию",
        data=gif_bytes,
        file_name="vortex_evolution.gif",
        mime="image/gif",
    )

# Настройка выбора режима граничных условий (стационарный или с излучающими границами Зоммерфельда) и генерация 6 GIF-анимаций для частиц
gif_bc_mode = st.radio(
    "Граничные условия для 6 GIF-анимаций:",
    [
        "1. Стационарное удержание (как сейчас: вязкая релаксация + фазовая волна)",
        "2. Чистое решение уравнений (излучающие границы Зоммерфельда, damping = 0 → распад мюона)",
    ],
    horizontal=True,
)
is_radiative = gif_bc_mode.startswith("2.")
if make_6_gifs:
    gifs_6 = []
    prog_6 = st.progress(0, text="Генерация GIF для 6 частиц...")
    cross_media = st.session_state.get("cross_media", {})

    for idx, (p_name, (m_p, n_p), m_targ, r_targ) in enumerate(FOUND_PARTICLES_BENCHMARK):
        prog_6.progress(idx / 6.0, text=f"Расчёт и рендер GIF ({idx + 1}/6): {p_name} ({m_p}, {n_p})...")
        scale_r = (2.0 * 3.0) / max(float(m_p * n_p), 1.0)
        r0_p = float(R_E_COMPTON_FM) * scale_r
        N_eff = min(max(int(N_grid), int(2 * max(abs(m_p), abs(n_p)))), 48)
        b_scale = 5.5 if is_radiative else 3.2

        grid_p = build_3d_grid(r0_p, N=N_eff, box_scale=b_scale)
        psi0_p = init_hopf_vortex_3d(grid_p, r0_p, r0_p * 0.30, int(m_p), int(n_p))
        med_p = cross_media.get(p_name, medium)

        res_p = relax_vortex_3d(
            psi0_p, grid_p, med_p,
            max_steps=600 if is_radiative else min(int(max_steps), 150),

            renormalize=False,
            damping_coeff=0.0 if is_radiative else 0.02,
            bc_mode="radiative" if is_radiative else "confined",
            record_frames=int(n_snap_frames) if is_radiative else 0,
            stop_on_buckling=not is_radiative,
        )
        if is_radiative and len(res_p.get("psi_frames", [])) >= 2:
            gif_b = build_vortex_evolution_gif(
                res_p["psi_frames"], grid_p,
                azim_deg=azim_deg, elev_deg=elev_deg, zoom=zoom_val,
            )
        else:
            gif_b = build_vortex_gif(
                res_p["psi"], grid_p,
                azim_deg=azim_deg, elev_deg=elev_deg,
                zoom=zoom_val, n_frames=int(n_snap_frames),
            )
        gifs_6.append((p_name, f"({m_p}, {n_p})", gif_b))
    prog_6.progress(1.0, text="Все 6 GIF-анимаций готовы!")
    st.session_state["vortex_gifs_6"] = gifs_6



if "vortex_gifs_6" in st.session_state:
    st.subheader("Динамика 6 найденных частиц")
    cols_6 = st.columns(3)
    for idx, (p_name, topo_str, gif_b) in enumerate(st.session_state["vortex_gifs_6"]):
        with cols_6[idx % 3]:
            st.markdown(f"**{p_name}** `{topo_str}`")
            st.image(gif_b)
            st.download_button(
                label=f"Скачать GIF ({topo_str})",
                data=gif_b,
                file_name=f"particle_{idx + 1}_{topo_str.strip('()').replace(', ', '_')}.gif",
                mime="image/gif",
                key=f"dl_gif_6_{idx}",
            )
