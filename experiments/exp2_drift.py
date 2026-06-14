import numpy as np
import matplotlib.pyplot as plt

from controllers.deepc_vanilla import VanillaDeePC
from controllers.deepc_online import OnlineDeePC
from controllers.closed_loop_driver import run_closed_loop_drift
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

err_vanilla = run_closed_loop_drift(vanilla, figure_eight)
err_online = run_closed_loop_drift(online, figure_eight)

t = np.arange(len(err_vanilla)) * 0.05

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

drift_idx = 200

rmse_vanilla = np.sqrt(np.mean(err_vanilla[drift_idx:] ** 2))
rmse_online = np.sqrt(np.mean(err_online[drift_idx:] ** 2))

print(f"Vanilla post-drift RMSE: {rmse_vanilla:.3f} m")
print(f"Online  post-drift RMSE: {rmse_online:.3f} m")