import numpy as np
import time
from sim.bicycle import KinematicBicycle, DriftingBicycle
from sim.references import double_lane_change


def run_closed_loop(controller, ref_func, T=20.0, dt=0.05,
                    n=6, L=24, noise=True):

    veh = KinematicBicycle(
        dt=dt,
        noise_std={'X': 0.005, 'Y': 0.005, 'psi': 0.01} if noise else None
    )

    veh.reset(x0=[0.0, 0.0, 0.0, 1.0])

    # Seed past buffer with n open-loop samples
    u_buf, y_buf = [], []

    for _ in range(n):
        u_init = np.array([0.0, 0.3])
        y_buf.append(veh.step(u_init))
        u_buf.append(u_init)

    log = {
        't': [],
        'y': [],
        'u': [],
        'ref': [],
        'ct': []
    }

    steps = int(T / dt)

    for k in range(n, steps):
        u_past = np.array(u_buf[-n:]).flatten()
        y_past = np.array(y_buf[-n:]).flatten()

        t_now = k * dt

        y_ref = np.array([
            ref_func(t_now + i * dt)
            for i in range(L)
        ])

        tic = time.perf_counter()

        u = controller(u_past, y_past, y_ref)

        ct = time.perf_counter() - tic
        
        y = veh.step(u)

        if hasattr(controller, "push_measurement"):
            controller.push_measurement(u, y)
       
        u_buf.append(u)
        y_buf.append(y)

        log['t'].append(t_now)
        log['y'].append(y)
        log['u'].append(u)
        log['ref'].append(y_ref[0])
        log['ct'].append(ct)

    return {k: np.array(v) for k, v in log.items()}

def run_closed_loop_drift(controller, ref_func,
                          T=20.0, dt=0.05,
                          n=6, L=24):

    veh = DriftingBicycle(
        dt=dt,
        drift_step=200,
        drift_factor=0.7
    )

    veh.reset(x0=[0.0, 0.0, 0.0, 1.0])

    # Seed past buffer
    u_buf, y_buf = [], []

    for _ in range(n):
        u_init = np.array([0.0, 0.3])

        y = veh.step(u_init)

        u_buf.append(u_init)
        y_buf.append(y)

    log = {
        't': [],
        'y': [],
        'u': [],
        'ref': [],
        'ct': [],
        'error': []
    }

    steps = int(T / dt)

    for k in range(n, steps):

        u_past = np.array(u_buf[-n:]).flatten()
        y_past = np.array(y_buf[-n:]).flatten()

        t_now = k * dt

        y_ref = np.array([
            ref_func(t_now + i * dt)
            for i in range(L)
        ])

        tic = time.perf_counter()

        u = controller(u_past, y_past, y_ref)

        ct = time.perf_counter() - tic

        y = veh.step(u)

        if hasattr(controller, "push_measurement"):
            controller.push_measurement(u, y)

        u_buf.append(u)
        y_buf.append(y)

        error = abs(y[1] - y_ref[0, 1])

        log['t'].append(t_now)
        log['y'].append(y)
        log['u'].append(u)
        log['ref'].append(y_ref[0])
        log['ct'].append(ct)
        log['error'].append(error)

    return {k: np.array(v) for k, v in log.items()}
