# DDMPC

## Execution
### Setup
python3 -m venv deepc-env
.\deepc-env\Scripts\activate (for windows)
pip install -r requirements.txt

### Experiments execution
<!-- execute from root folder -->
<!-- Do not add .py extension to the experiment file -->
python3 -m experiments.<experiment file>

## Controllers

### VanillaDeePC
Baseline Data-Enabled Predictive Control implementation (`deepc_vanilla.py`). Takes offline input/output data (`U_data`, `Y_data`), builds Hankel matrices of order `L + n` once at construction time, and splits each Hankel matrix into a "past" block (first `n` row-blocks) and a "future" block (next `L` row-blocks). The past blocks enforce that the trajectory implied by `alpha` matches recently observed input/output history; the future blocks are constrained to equal the predicted `u_bar`/`y_bar` over the horizon. The QP minimizes a regularized least-squares cost (`lam * ||alpha||^2` plus tracking and control-effort terms weighted by `Q_sqrt`/`R_sqrt`), subject to input bounds.

`_build_problem()` is factored out as its own method specifically so that subclasses can override it without re-implementing the Hankel construction in `__init__`. The QP is solved with OSQP and warm-started; calling the controller (`__call__`) updates the `u_past`, `y_past`, and `y_ref` parameters, solves, and returns the first control action `u_bar[0]`. If the solver does not return `OPTIMAL` or `OPTIMAL_INACCURATE`, it raises `RuntimeError` rather than silently returning a stale/invalid action — this is what experiment scripts catch when sweeping parameters.

### OnlineDeePC
Sliding-window extension of `VanillaDeePC` (`deepc_online.py`). Keeps `U_buf`/`Y_buf` as running lists seeded from the offline data, and `push_measurement()` appends new (u, y) pairs at each control step, popping from the front once `window_size` is exceeded. Every `update_every` steps, `_rebuild_hankel()` reconstructs `Hu`, `Hy` (and their past/future splits) from the current buffer and calls `_build_problem()` again, so the QP variables and constraints are rebuilt from scratch with the new data.

This is the mechanism intended to let the controller adapt to plant drift (e.g., a changing bicycle model) without needing new offline data collection. Note that `_rebuild_hankel()` does not update `self.prob` incrementally — it fully reconstructs the cvxpy problem each time, so this is more expensive per rebuild than a single solve; `update_every` is the main lever for trading off adaptation speed against compute cost.

### ReducedOrderDeePC
SVD-reduced extension of `VanillaDeePC` (`deepc_reduced.py`). Overrides `_build_problem()` to perform an SVD-based dimensionality reduction on the stacked Hankel matrix `[Hu; Hy]` before formulating the QP. Two selection modes are supported, controlled by which constructor argument is set: `rank` keeps a fixed number of singular vectors `k = min(rank, len(s))`; `energy_ratio` instead picks the smallest `k` such that cumulative singular-value energy reaches the target ratio, with a floor of `k_min = n * (m + p)` (this floor exists because fewer columns than that are not enough to pin down the past trajectory uniquely, which can make the problem rank-deficient).

The retained right-singular vectors `V_r` are used to project `Hu_p`, `Hy_p`, `Hu_f`, `Hy_f` into the reduced space, and the QP variable becomes `alpha_r` of dimension `k` instead of the full column count of the Hankel matrix. This trades solution fidelity for solve speed; the `exp3_pareto_*` experiments exist to characterize that tradeoff empirically. Because `_build_problem()` is called from the parent `__init__`, and depends on `self.Hu`/`self.Hy` already being set, the SVD step only ever sees the matrices built by `VanillaDeePC.__init__` — `ReducedOrderDeePC` does not currently support being combined with `OnlineDeePC`'s rebuild mechanism; doing so would require also overriding `_rebuild_hankel()` or refactoring shared logic.

### mpc
Linear time-varying MPC for a kinematic bicycle model (`mpc_v2.py`), used as a baseline/comparison against the DeePC family. At each `solve(x0, t0, u_prev)` call it generates a reference trajectory over the horizon via `reference_func` (default `figure_eight`), computes a reference control sequence by inverting the bicycle kinematics (`_reference_input`), and linearizes the dynamics around each reference point (`_linearize`) to produce a time-varying `A_i, B_i, c_i` affine model for each step of the horizon.

Two details are easy to miss when debugging tracking error. First, the reference heading `psi_ref` from `figure_eight` is unwrapped (continuous, can exceed ±π), so the code explicitly avoids `wrap_angle()` when computing `psi_dot` from consecutive reference points — wrapping there would corrupt the yaw-rate estimate across multi-revolution trajectories. Second, before optimization the vehicle's current yaw `x0[2]` is re-aligned to the nearest 2π-multiple of the reference yaw branch (`psi0_aligned`), so that the linearization error term `x - x_ref` doesn't get artificially inflated by a 2π wraparound mismatch between vehicle yaw (typically wrapped) and reference yaw (unwrapped).

The QP is solved with OSQP, falling back to ECOS on solver exceptions; if both fail (`x.value is None`), it returns a zero control rather than raising, which is different from the DeePC controllers' fail-fast behavior — worth knowing if an experiment appears to silently stall instead of crashing.

## Experiments

All experiment scripts are run as modules from the project root (`python3 -m experiments.<name>`, without the `.py` extension) and expect `data/open_loop.npz` (containing arrays `U` and `Y`) to exist beforehand, except `experiment_1_figure_eight.py` which runs the MPC controller against a simulated plant directly and does not need offline data.

### exp1_vanilla
Runs `VanillaDeePC` against a double-lane-change reference for 32 seconds with no added noise (`noise=False`), via `run_closed_loop`. Plots the XY trajectory against the reference and prints mean solve time plus mean/total/max/RMSE tracking error. This is the simplest script in the set and is a good first sanity check that the Hankel data, controller construction, and closed-loop driver are wired up correctly before running anything more complex.

### exp2_drift / exp2_drift2
Compare `VanillaDeePC` against `OnlineDeePC` (with `update_every=10`) under a drifting plant model (`run_closed_loop_drift`, using `DriftingBicycle`), on `figure_eight` (`exp2_drift.py`) and `double_lane_change` (`exp2_drift2.py`) references respectively. Both hardcode `drift_idx = 200` as the sample index at which drift is assumed to begin (annotated on the time-series plot as `200 * 0.05` seconds), and compute post-drift RMSE only over samples after that index — if the drift onset timing in `run_closed_loop_drift` or the simulation `dt` changes, this index needs to be updated by hand in both scripts, since it is not derived from the actual drift schedule. Each script saves a lateral-error-over-time plot and an XY trajectory comparison plot to `results/`.

### exp2_no_drift
The non-drift control case for the same comparison (`VanillaDeePC` vs `OnlineDeePC`, `figure_eight` reference, via `run_closed_loop` rather than the drift variant). Intended to confirm that `OnlineDeePC`'s periodic rebuilding does not itself degrade performance when there is nothing to adapt to. Note it still slices and reports RMSE from `drift_idx = 200` onward even though there is no drift in this script — this looks like a copy-paste artifact from `exp2_drift.py` rather than intentional, so treat the printed numbers as "RMSE over the back half of the run" rather than "post-drift RMSE."

### exp3_pareto_rank / exp3_pareto_sweep
Characterize the accuracy/speed tradeoff of `ReducedOrderDeePC` by sweeping, respectively, a list of fixed `rank` values (`30` to `150`) and a list of `energy_ratio` values (`0.80` to `1.00`), each run against `double_lane_change` for 20 seconds. For each setting they rebuild a fresh controller, run the closed loop, and record RMSE and mean per-step solve time, catching `RuntimeError` and `SolverError` so a failed configuration is logged as `NaN` rather than aborting the sweep. Results are plotted as solve-time vs. RMSE scatter plots with point labels, intended to be read as Pareto frontiers — a configuration is only interesting if no other point dominates it on both axes. If a particular rank or energy ratio fails to solve, check the console output printed before the `NaN` is appended; it carries the original exception message.

### exp4_all
The consolidated experiment: loops over a `references` dict (currently `double_lane_change` and `figure_eight`; `straight_line`, `circle`, and `lane_change` are present but commented out) and, for each reference, runs all three controllers (`VanillaDeePC`, `OnlineDeePC`, `ReducedOrderDeePC` with `REDUCED_RANK = 140`) under both no-drift and drift conditions, producing trajectory plots, error-over-time plots, and a printed min/max/avg/RMSE error table per condition. Controllers are rebuilt fresh per condition via `build_controllers()` so that state from the no-drift run (e.g. `OnlineDeePC`'s buffer) does not leak into the drift run.

One thing to watch in `make_plot()`: the call signature takes `vanilla_log` twice (once positionally as `ref_log` and again as `vanilla_log`) — this works because the reference trajectory is read off of `vanilla_log["ref"]`, but it means the reference line plotted is always taken from the vanilla run's log, not from a reference-only source; this is fine as long as all three controllers are tracking the same reference, but would silently mislabel the reference curve if that assumption is ever broken. This is the script to run for the most complete picture before adding a new controller variant or reference trajectory, since it exercises every existing combination in one pass.

### experiment_1_figure_eight
The MPC-only experiment, separate from the DeePC family. Instantiates `mpc` with a figure-eight reference (`A=10.0`, `omega=0.1`), tuned cost matrices, and a 25-step horizon, then closes the loop manually (no shared `closed_loop_driver` helper) against a `KinematicBicycle` plant for 64 seconds. It manipulates `sys.path` directly at the top of the file to ensure the project root is importable, which suggests it may have originally been written to be run as a standalone script rather than as a module (`python3 -m experiments.experiment_1_figure_eight`) — if imports fail when run the standard way, check whether this path-manipulation block is interfering with or duplicating module resolution. Saves both a trajectory plot and a position-error-over-time plot, and prints mean/max tracking error to console.