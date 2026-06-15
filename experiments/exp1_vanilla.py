import numpy as np
import matplotlib.pyplot as plt

from controllers.deepc_vanilla import VanillaDeePC
from sim.references import double_lane_change
from sim.closed_loop_driver import run_closed_loop

data = np.load('data/open_loop.npz')

ctrl = VanillaDeePC(
    data['U'],
    data['Y'],
    L=24,
    n=6,
    lam=1e-3
)

log = run_closed_loop(
    ctrl,
    double_lane_change,
    T=20.0
)

plt.figure(figsize=(10, 4))

plt.plot(
    log['ref'][:, 0],
    log['ref'][:, 1],
    'k--',
    label='reference'
)

plt.plot(
    log['y'][:, 0],
    log['y'][:, 1],
    'r-',
    label='vanilla DeePC'
)

plt.axis('equal')
plt.legend()
plt.xlabel('X (m)')
plt.ylabel('Y (m)')

plt.title('Day 3 -- Vanilla DeePC, Double Lane Change')

plt.savefig(
    'results/day3_vanilla_dlc.png',
    dpi=150
)

print(f"Mean solve time: {1000 * log['ct'].mean():.1f} ms")

err = np.linalg.norm(
    log['y'][:, :2] - log['ref'][:, :2],
    axis=1
)

print(f"Mean tracking error: {err.mean():.4f} m")
print(f"Total tracking error: {err.sum():.4f} m")
print(f"Max tracking error: {err.max():.4f} m")
print(f"RMSE tracking error: {np.sqrt(np.mean(err**2)):.4f} m")