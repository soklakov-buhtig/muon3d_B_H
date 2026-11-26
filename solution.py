import numpy as np
from matrix_assembly import evaluate_energy_and_gradient


def _apply_sommerfeld_bc(psi_new, psi_prev, cfl_c=0.45):
    """Излучающие граничные условия Зоммерфельда 1-го порядка на 6 гранях куба."""
    r = float(cfl_c)
    psi_new[0, :, :] = psi_prev[0, :, :] + r * (psi_prev[1, :, :] - psi_prev[0, :, :])
    psi_new[-1, :, :] = psi_prev[-1, :, :] + r * (psi_prev[-2, :, :] - psi_prev[-1, :, :])
    psi_new[:, 0, :] = psi_prev[:, 0, :] + r * (psi_prev[:, 1, :] - psi_prev[:, 0, :])
    psi_new[:, -1, :] = psi_prev[:, -1, :] + r * (psi_prev[:, -2, :] - psi_prev[:, -1, :])
    psi_new[:, :, 0] = psi_prev[:, :, 0] + r * (psi_prev[:, :, 1] - psi_prev[:, :, 0])
    psi_new[:, :, -1] = psi_prev[:, :, -1] + r * (psi_prev[:, :, -2] - psi_prev[:, :, -1])
    return psi_new


def relax_vortex_3d(
    psi_init,
    grid,
    medium,
    max_steps=600,
    tol_energy=1e-6,
    renormalize=False,
    target_amp=1.0,
    norm_interval=1,
    damping_coeff=0.02,
    bc_mode="confined",
    record_frames=0,
    stop_on_buckling=True,
):
    """
    Релаксация 3D-поля psi(x, y, z) по схеме leapfrog + вязкость с шагом dt = tau / 2.
    """
    tau = float(medium.get("tau", 1.0))
    dt = 0.5 * tau
    damping = float(damping_coeff)
    use_mask = (bc_mode == "confined")

    damping = float(damping_coeff)


    dx = grid["dx"]
    mask = grid["boundary_mask"]

    # Масштабный множитель для согласования размерности dE/dpsi с шагом сетки dx
    # Гарантирует выполнение условия Куранта (CFL) при любом R0_fm
#    cfl_scale = (dx ** 2) / 200.0
    # Адаптивная нормировка шага по пиковому градиенту (исключает взрыв и сдувание при любом dx)


    psi = psi_init.copy().astype(np.complex128)
    psi_old = psi.copy()

    energies, dE_dpsi = evaluate_energy_and_gradient(psi, grid, medium, use_mask=use_mask)

#    energies, dE_dpsi = evaluate_energy_and_gradient(psi, grid, medium, use_mask=use_mask)
    history = [
        {
            "step": 0,
            "E_total": energies["E_total"],
            "E1": energies["E1"],
            "E2": energies["E2"],
            "E3": energies["E3"],
            "E4": energies["E4"],
            "max_amp": float(np.max(np.abs(psi))),
            "chi_max": energies.get("chi_max", 0.0),
        }
    ]

    prev_E = energies["E_total"]
    converged = False
    buckling_broken = False
    use_mask = (bc_mode == "confined")
    psi_frames = [psi.copy()] if int(record_frames) > 0 else []
    snap_every = max(1, int(max_steps) // max(int(record_frames), 1))

    for step in range(1, int(max_steps) + 1):
        max_grad = float(np.max(np.abs(dE_dpsi)))
#        cfl_scale = 0.05 / max(max_grad, 1e-12)
#        cfl_scale = 0.005 / max(max_grad, 1e-12)
        cfl_scale = (0.005 if use_mask else 0.02) / max(max_grad, 1e-12)


        force = -dE_dpsi * cfl_scale
    
                # Шаг leapfrog с демпфированием
        psi_new = (
            2.0 * psi
            - psi_old
            + (dt ** 2) * force
            - damping * (psi - psi_old)
        ) / (1.0 + damping)

        # Применение открытых граничных условий (open BC: поле -> 0 на границе)
        if use_mask:
            psi_new *= mask
        else:
            psi_new = _apply_sommerfeld_bc(psi_new, psi, cfl_c=min(0.45, 0.25 * dt))


        # Периодическая перенормировка от численного сдувания (каждые norm_interval шагов)
        if renormalize and norm_interval > 0 and (step % int(norm_interval) == 0):
            cur_max = float(np.max(np.abs(psi_new)))
            if cur_max > 1e-15:
                scale = float(target_amp) / cur_max
                psi_new *= scale
                psi *= scale

        psi_old = psi
        psi = psi_new
        if int(record_frames) > 0 and (step % snap_every == 0) and len(psi_frames) < int(record_frames):
            psi_frames.append(psi.copy())


        # Исправлен вызов функции с передачей use_mask
        energies, dE_dpsi = evaluate_energy_and_gradient(psi, grid, medium, use_mask=use_mask)
        cur_E = energies["E_total"]
        # ... (основной цикл расчёта с проверками сходимости и энергии)
        if step % 10 == 0 or step == max_steps:
            history.append({
                "step": step, "E_total": cur_E,
                "E1": energies["E1"], "E2": energies["E2"],
                "E3": energies["E3"], "E4": energies["E4"],
                "max_amp": float(np.max(np.abs(psi))),
            })

        if not np.isfinite(cur_E):
            break
        if stop_on_buckling and not energies.get("buckling_ok", True):
            buckling_broken = True
            break
        if stop_on_buckling and float(np.max(np.abs(psi))) < 1e-4:
            break
        prev_E = cur_E
    amp_final = float(np.max(np.abs(psi)))
    amp_init = max(float(np.max(np.abs(psi_init))), 1e-12)
    amp_ratio = amp_final / amp_init
    norm_init = max(float(np.sum(np.abs(psi_init) ** 2)), 1e-12)
    norm_final = float(np.sum(np.abs(psi) ** 2))
    norm_ratio = norm_final / norm_init
    dynamically_stable = bool(
        (not buckling_broken)
        and np.isfinite(cur_E)
        and (abs(cur_E) > 1e-4)
        and (1e-4 <= norm_ratio <= 20.0)
        and (step == int(max_steps))
    )
    return {
        "psi": psi, "energies": energies, "history": history,
        "converged": dynamically_stable, "dynamically_stable": dynamically_stable,
        "buckling_ok": not buckling_broken, "amp_ratio": amp_ratio,
        "steps_done": step, "dt": dt, "tau": tau, "psi_frames": psi_frames,
    }
