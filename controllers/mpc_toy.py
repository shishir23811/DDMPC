import numpy as np
from itertools import product

np.random.seed(0)

N = 3

steering_vals = np.deg2rad(np.linspace(-20, 20, 10))

# accel_vals = np.linspace(-5.0, 5.0, 20)
accel_vals = [-5.0, 0.0, 5.0]
# accel_vals = [0.0]

def predict_step(state, u, wheel_base=0.3, dt=0.05):
    delta, a = u

    X, Y, psi, v = state

    X_n = X + v * np.cos(psi) * dt
    Y_n = Y + v * np.sin(psi) * dt
    psi_n = psi + (v / wheel_base) * np.tan(delta) * dt
    v_n = max(0.0, v + a * dt)

    return np.array([X_n, Y_n, psi_n, v_n])

def mpc_control(state, refs):
    best_cost = np.inf
    best_u = np.array([0.0, 0.0])
    errors = []


    for steer_seq in product(steering_vals, repeat=N):

        for accel_seq in product(accel_vals, repeat=N):

            x = state.copy()
            cost = 0.0
            errors_local = []
            num_samples = 1000

            # for _ in range(num_samples):

            #     steer_seq = np.random.choice(
            #         steering_vals,
            #         size=N
            #     )

            #     accel_seq = np.random.choice(
            #         accel_vals,
            #         size=N
            #     )

            x = state.copy()
            cost = 0.0
            for k in range(N):

                u = np.array([
                    steer_seq[k],
                    accel_seq[k]
                ])

                x = predict_step(x, u)

                ref = refs[k]

                ex = x[0] - ref[0]
                ey = x[1] - ref[1]

                epsi = np.arctan2(
                    np.sin(x[2] - ref[2]),
                    np.cos(x[2] - ref[2])
                )
                
                cost += (
                    9.0 * ex**2 +
                    18.0 * ey**2 +
                    5.0 * epsi**2 +
                    0.1 * u[0]**2 + 0.7 * u[1]**2
                )
                if k == 0:
                    errors_local = [ex, ey, epsi]


            if cost < best_cost:

                best_cost = cost
                best_u = np.array([
                    steer_seq[0],
                    accel_seq[0]
                ])
                errors = errors_local.copy()

    return best_u, errors