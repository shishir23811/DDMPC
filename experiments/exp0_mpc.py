import numpy as np
import matplotlib.pyplot as plt

from sim.bicycle import KinematicBicycle
from sim.references import figure_eight
from controllers.mpc import mpc_control, N

sim = KinematicBicycle()

T = 640

x_hist = []
y_hist = []

ref_x_hist = []
ref_y_hist = []

total_errors= []
for t in range(T):

    refs = []

    for k in range(N):

        ref = figure_eight(
            (t + k + 1) * sim.dt
        )

        refs.append(ref)

    u, errors = mpc_control(
        sim.state,
        refs
    )

    sim.step(u)

    x_hist.append(sim.state[0])
    y_hist.append(sim.state[1])

    ref_now = figure_eight(t * sim.dt)

    ref_x_hist.append(ref_now[0])
    ref_y_hist.append(ref_now[1])
    total_errors.append(errors)


total_errors = np.array(total_errors)

mean_err = np.mean(np.abs(total_errors), axis=0)
max_err = np.max(np.abs(total_errors), axis=0)
rmse = np.sqrt(np.mean(total_errors**2, axis=0))

print("Mean :", mean_err)
print("Max  :", max_err)
print("RMSE :", rmse)

tol_x = 0.5      
tol_y = 0.5      
tol_yaw = 0.1    

tol = np.array([tol_x, tol_y, tol_yaw])

normalized_rmse = rmse / tol

print(normalized_rmse)

plt.figure(figsize=(8, 6))

plt.plot(
    ref_x_hist,
    ref_y_hist,
    '--',
    linewidth=2,
    label='Reference'
)

plt.plot(
    x_hist,
    y_hist,
    linewidth=2,
    label='MPC'
)

plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('Figure-8')
plt.axis('equal')
plt.grid(True)
plt.legend()

plt.show()