import os
import numpy as np
import matplotlib.pyplot as plt

from controllers.deepc_vanilla import VanillaDeePC
from controllers.deepc_online import OnlineDeePC
from controllers.deepc_reduced import ReducedOrderDeePC

from sim.closed_loop_driver import (
    run_closed_loop,
    run_closed_loop_drift
)

from sim.references import (
    straight_line,
    circle,
    lane_change,
    double_lane_change,
    figure_eight
)

# =====================================================
# Configuration
# =====================================================

L = 24
n = 6
lam = 1e-3

REDUCED_RANK = 140

T_NORMAL = 32.0
T_DRIFT = 32.0

os.makedirs("results", exist_ok=True)

# =====================================================
# Load data
# =====================================================

data = np.load("data/open_loop.npz")

# =====================================================
# References
# =====================================================

references = {
    # "straight_line": straight_line,
    # "circle": circle,
    # "lane_change": lane_change,
    "double_lane_change": double_lane_change,
    "figure_eight": figure_eight,
}

def print_errors(ref_name, condition, logs: dict):
    """Print min/max/avg/RMSE position errors for each controller."""

    print(f"\n{ref_name.replace('_', ' ').title()} — {condition}")
    print(f"  {'Controller':<18} {'Min':>7} {'Max':>7} {'Avg':>7} {'RMSE':>7}")
    print(f"  {'-'*46}")

    for name, log in logs.items():
        errs = np.linalg.norm(log["y"][:, :2] - log["ref"][:, :2], axis=1)
        rmse = np.sqrt(np.mean(errs ** 2))
        print(f"  {name:<18} {errs.min():>6.3f}m {errs.max():>6.3f}m {errs.mean():>6.3f}m {rmse:>6.3f}m")

def make_error_plot(
    filename,
    title,
    vanilla_log,
    online_log,
    reduced_log,
    dt=0.05
):
    plt.figure(figsize=(8, 3))

    logs = {
        "Vanilla DeePC": (vanilla_log, "r"),
        "Online DeePC": (online_log, "b"),
        "Reduced DeePC": (reduced_log, "g"),
    }

    for name, (log, color) in logs.items():

        errors = np.linalg.norm(
            log["y"][:, :2] - log["ref"][:, :2],
            axis=1
        )

        t = np.arange(len(errors)) * dt

        plt.plot(
            t,
            errors,
            color=color,
            label=name
        )

    plt.axhline(
        0.1,
        color="k",
        linestyle="--",
        label="0.1 m target"
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Position Error [m]")
    plt.title(title)

    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join("results", filename),
        dpi=150
    )

    plt.close()

# =====================================================
# Controller factory
# =====================================================

def build_controllers():

    vanilla = VanillaDeePC(
        data["U"],
        data["Y"],
        L=L,
        n=n,
        lam=lam
    )

    online = OnlineDeePC(
        data["U"],
        data["Y"],
        L=L,
        n=n,
        lam=lam,
        update_every=10
    )

    reduced = ReducedOrderDeePC(
        data["U"],
        data["Y"],
        rank=REDUCED_RANK,
        L=L,
        n=n,
        lam=lam
    )

    return vanilla, online, reduced

# =====================================================
# Plot helper
# =====================================================

def make_plot(
    filename,
    title,
    ref_log,
    vanilla_log,
    online_log,
    reduced_log
):

    plt.figure(figsize=(9, 6))

    plt.plot(
        ref_log["ref"][:, 0],
        ref_log["ref"][:, 1],
        "k--",
        linewidth=2,
        label="Reference"
    )

    plt.plot(
        vanilla_log["y"][:, 0],
        vanilla_log["y"][:, 1],
        "r",
        label="Vanilla DeePC"
    )

    plt.plot(
        online_log["y"][:, 0],
        online_log["y"][:, 1],
        "b",
        label="Online DeePC"
    )

    plt.plot(
        reduced_log["y"][:, 0],
        reduced_log["y"][:, 1],
        "g",
        label="Reduced DeePC"
    )

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")

    plt.title(title)

    plt.axis("equal")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join("results", filename),
        dpi=150
    )

    plt.close()

# =====================================================
# Run experiments
# =====================================================

for ref_name, ref_func in references.items():

    print(f"\n{'='*60}")
    print(f"Reference: {ref_name}")
    print(f"{'='*60}")

    # -------------------------------------------------
    # NO DRIFT
    # -------------------------------------------------

    print("Running no-drift experiment...")

    vanilla, online, reduced = build_controllers()

    log_vanilla = run_closed_loop(
        vanilla,
        ref_func,
        T=T_NORMAL,
        noise=False,
        n=n,
        L=L
    )

    log_online = run_closed_loop(
        online,
        ref_func,
        T=T_NORMAL,
        noise=False,
        n=n,
        L=L
    )

    log_reduced = run_closed_loop(
        reduced,
        ref_func,
        T=T_NORMAL,
        noise=False,
        n=n,
        L=L
    )

    make_plot(
        f"{ref_name}_no_drift.png",
        f"{ref_name.replace('_',' ').title()} (No Drift)",
        log_vanilla,
        log_vanilla,
        log_online,
        log_reduced
    )

    make_error_plot(
    f"{ref_name}_no_drift_error.png",
    f"{ref_name.replace('_',' ').title()} Error (No Drift)",
    log_vanilla,
    log_online,
    log_reduced
    )

    print_errors(ref_name, "No Drift", {
        "Vanilla":  log_vanilla,
        "Online":   log_online,
        "Reduced":  log_reduced,
    })

    print(
        f"Saved results/{ref_name}_no_drift.png"
    )

    # -------------------------------------------------
    # DRIFT
    # -------------------------------------------------

    print("Running drift experiment...")

    vanilla, online, reduced = build_controllers()

    log_vanilla = run_closed_loop_drift(
        vanilla,
        ref_func,
        T=T_DRIFT,
        n=n,
        L=L
    )

    log_online = run_closed_loop_drift(
        online,
        ref_func,
        T=T_DRIFT,
        n=n,
        L=L
    )

    log_reduced = run_closed_loop_drift(
        reduced,
        ref_func,
        T=T_DRIFT,
        n=n,
        L=L
    )

    make_plot(
        f"{ref_name}_drift.png",
        f"{ref_name.replace('_',' ').title()} (Drift)",
        log_vanilla,
        log_vanilla,
        log_online,
        log_reduced
    )

    make_error_plot(
    f"{ref_name}_drift_error.png",
    f"{ref_name.replace('_',' ').title()} Error (Drift)",
    log_vanilla,
    log_online,
    log_reduced
    )

    print_errors(ref_name, "Drift", {
        "Vanilla":  log_vanilla,
        "Online":   log_online,
        "Reduced":  log_reduced,
    })

    print(
        f"Saved results/{ref_name}_drift.png"
    )