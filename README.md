# muon3d
Only full 3d with dt=tau/2 can solve the e1-e4

# 3D Topological Soliton Simulator (`muon3d_B_H`)

An open-source 3D numerical and analytical simulation framework exploring elementary particles as topologically knotted electromagnetic solitons `(p, q)` in a continuous, nonlinear elasto-plastic vacuum medium.

---

## Core Physical Idea

Operating strictly within flat 3D Euclidean space and absolute uniform time, the model investigates whether localized particle states (such as the electron, muon, and light mesons) can emerge as self-confined toroidal–poloidal Hopf vortices governed solely by nonlinear Maxwell–Plebanski field equations with Born–Infeld constitutive saturation ($\varepsilon_r = 1/\Gamma$, $\mu_r = \Gamma$) and zero external sources ($\rho_e \equiv 0$, $\mathbf{j} \equiv 0$).

---

## What Is Currently Implemented & Observed

1. **Analytical Scaling & Spectrum:**
   * Derives characteristic mass scales and radii from topological winding numbers `(p, q)` without introducing external point charges or probabilistic wavefunctions.
   * Establishes a qualitative hierarchy between the ground-state `(2, 3)` mode (electron) and higher-winding excitations (`(19, 31)`, `(27, 43)`, etc.).

2. **3D Leapfrog Field Solver (`solution.py`):**
   * Integrates the coupled nonlinear field equations on a 3D Cartesian grid under two boundary regimes:
     * **Confined / Damped Relaxation:** Isolates stationary equilibrium soliton profiles via artificial viscosity/damping.
     * **Radiative Open Boundaries (Sommerfeld BC):** Tracks free dynamical evolution and shear-buckling decay.

3. **3D Flux Ribbon & Topological Center Visualization (`ribbon_renderer.py`, `make_animation.py`):**
   * Reconstructs 3D $\mathbf{B} \times \mathbf{H}$ energy-flux ribbons with phase twist and tracks energy-weighted connected components (3D triple crosshairs) to visualize vortex core integrity versus fragmentation.

---

## Honest Limitations & Open Problems (Current Status)

To maintain strict scientific transparency, the current numerical implementation has the following known boundaries:

* **Derrick-Type Dispersion in Pure Undamped Evolution:** Under strictly zero viscosity (`eta_visc = 0`, `damping = 0`) and open Sommerfeld boundaries, localized 3D electromagnetic configurations eventually disperse or undergo shear-buckling fragmentation over long integration times unless stabilized by a medium viscosity term or an additional dispersion/Skyrme–Faddeev-like topological barrier in the constitutive equations.
* **Grid Resolution Constraints for High Winding Numbers:** While low-winding modes like `(2, 3)` resolve cleanly on `48³–80³` grids, high-winding modes such as `(19, 31)` or `(29, 47)` contain closely spaced filament turns that suffer from numerical grid merging and discretization noise at moderate resolutions.
* **Visualization vs. Dynamics:** `matplotlib`-based 3D ribbon rendering provides a diagnostic view of field topology and fragmentation, but it is a coarse-grained sampling (10 seed trajectories) rather than a full volumetric field render.

---

## Repository Structure

* `streamlit_app.py` — Interactive web UI for running 3D simulations, adjusting medium/grid parameters, and exporting GIF animations.
* `constants.py` — Physical constants, topological benchmarks `(p, q)`, and reference PDG particle lifetimes.
* `preprocessor.py` — 3D Cartesian grid generation and initial Hopf-knotted field initialization.
* `solution.py` — CFL-adaptive 3D leapfrog solver with Born–Infeld nonlinearity and radiative/confined boundary conditions.
* `ribbon_renderer.py` — 3D $\mathbf{B} \times \mathbf{H}$ flux ribbon geometry and connected-component center detector.
* `make_animation.py` — Frame generation and GIF export utilities.

---

## Quick Start

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Feedback & Collaboration

Constructive critique, analytical insights regarding soliton stabilization terms in nonlinear electrodynamics, and suggestions on high-resolution numerical schemes are warmly welcomed via GitHub Issues or Discussions.
