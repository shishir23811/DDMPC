import numpy as np
import matplotlib.pyplot as plt

from cvxpy.error import SolverError

from controllers.deepc_reduced import ReducedOrderDeePC
from sim.closed_loop_driver import run_closed_loop
from sim.references import double_lane_change


data = np.load("data/open_loop.npz")

energy_ratios = [
    0.80,
    0.90,
    0.95,
    0.99,
    1.00
]

rmse_list = []
solve_time_list = []

for ratio in energy_ratios:

    print(f"\nRunning energy_ratio={ratio}")

    controller = ReducedOrderDeePC(
        data["U"],
        data["Y"],
        energy_ratio=ratio,
        L=24,
        n=6,
        lam=1e-3
    )

    try:

        log = run_closed_loop(
            controller,
            double_lane_change,
            T=20.0,
            dt=0.05,
            n=6,
            L=24
        )

    except (RuntimeError, SolverError) as e:

        print(e)

        rmse_list.append(np.nan)
        solve_time_list.append(np.nan)

        continue

    y = log["y"]
    ref = log["ref"]

    lateral_error = (
        y[:, 1] - ref[:, 1]
    )

    rmse = np.sqrt(
        np.mean(lateral_error**2)
    )

    mean_solve_time = np.mean(
        log["ct"]
    )

    rmse_list.append(rmse)
    solve_time_list.append(
        mean_solve_time
    )

    print(
        f"RMSE = {rmse:.4f} m, "
        f"Mean Solve Time = "
        f"{1000 * mean_solve_time:.2f} ms"
    )

plt.figure(figsize=(6, 4))

plt.scatter(
    rmse_list,
    np.array(solve_time_list) * 1000,
    s=80
)

for ratio, rmse, st in zip(
    energy_ratios,
    rmse_list,
    solve_time_list
):
    plt.annotate(
        f"{ratio:.2f}",
        (rmse, st * 1000)
    )

plt.xlabel("RMSE (m)")
plt.ylabel("Mean Solve Time (ms)")
plt.title("Energy-based Pareto Frontier")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/day5_pareto_energy.png",
    dpi=150
)