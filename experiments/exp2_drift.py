import numpy as np
import matplotlib.pyplot as plt

from controllers.deepc_vanilla import VanillaDeePC
from controllers.deepc_online import OnlineDeePC
from sim.closed_loop_driver import run_closed_loop_drift
from sim.bicycle import DriftingBicycle
from sim.references import figure_eight

data = np.load("data/open_loop.npz")

vanilla = VanillaDeePC(data["U"], data["Y"], L=24, n=6, lam=1e-3)

online = OnlineDeePC(
    data["U"],
    data["Y"],
    L=24,
    n=6,
    lam=1e-3,
    update_every=10
)

log_vanilla = run_closed_loop_drift(vanilla, figure_eight, T= 32.0)
log_online = run_closed_loop_drift(online, figure_eight, T= 32.0)

err_vanilla = np.abs(
    log_vanilla["y"][:, 1] -
    log_vanilla["ref"][:, 1]
)

err_online = np.abs(
    log_online["y"][:, 1] -
    log_online["ref"][:, 1]
)

t = log_vanilla["t"]

plt.figure(figsize=(10, 4))
plt.plot(t, err_vanilla, label="Vanilla DeePC")
plt.plot(t, err_online, label="Online DeePC")
plt.axvline(200 * 0.05, linestyle="--", label="Drift starts")

plt.xlabel("Time (s)")
plt.ylabel("Lateral Error (m)")
plt.title("Drift Comparison")
plt.legend()
plt.grid(True)

plt.savefig("results/day4_drift_comparison.png", dpi=150)

plt.figure(figsize=(8, 6))

plt.plot(
    log_vanilla["ref"][:, 0],
    log_vanilla["ref"][:, 1],
    "k--",
    label="Reference"
)

plt.plot(
    log_vanilla["y"][:, 0],
    log_vanilla["y"][:, 1],
    "r-",
    label="Vanilla DeePC"
)

plt.plot(
    log_online["y"][:, 0],
    log_online["y"][:, 1],
    "b-",
    label="Online DeePC"
)

plt.axis("equal")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.title("Trajectory Comparison Under Drift")
plt.legend()
plt.grid(True)

plt.savefig(
    "results/day4_drift_trajectory.png",
    dpi=150
)

drift_idx = 200

rmse_vanilla = np.sqrt(np.mean(err_vanilla[drift_idx:] ** 2))
rmse_online = np.sqrt(np.mean(err_online[drift_idx:] ** 2))

print(f"Vanilla post-drift RMSE: {rmse_vanilla:.3f} m")
print(f"Online  post-drift RMSE: {rmse_online:.3f} m")