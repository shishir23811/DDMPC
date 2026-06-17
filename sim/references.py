import numpy as np

def sig(x):
    return 1.0 / (1.0 + np.exp(-x))

def straight_line(t, v_long= 1.0):
    '''
    Vehicle moves along +X.
    Y is zero.
    '''

    X = v_long * t
    Y = 0.0
    psi = 0.0

    return np.array([X, Y, psi])

def circle(t, A= 5.0, omega= 0.2):
    '''
    discrete points circle
    Return:  [X, Y, psi]
    '''

    X = A * np.cos(omega * t)
    Y = A * np.sin(omega * t)

    dt_small = 1e-3

    X2 = A * np.cos(omega * (t + dt_small))
    Y2 = A * np.sin(omega * (t + dt_small))

    psi = np.arctan2(Y2 - Y, X2 - X)

    return np.array([X, Y, psi])

def figure_eight(t, A=5.0, omega=0.2):
    '''
    Analytical figure-8 reference path.
    Return: [X, Y, psi]
    '''

    X = A * np.sin(omega * t)
    Y = A * np.sin(2 * omega * t)

    t_val = float(t)
    if t_val <= 0.0:
        dX = A * omega * np.cos(0.0)
        dY = 2.0 * A * omega * np.cos(0.0)
        psi = np.arctan2(dY, dX)
        return np.array([X, Y, psi])

    n = max(int(t_val * 100) + 2, 4)
    t_grid = np.linspace(0.0, t_val, n)
    dX_grid = A * omega * np.cos(omega * t_grid)
    dY_grid = 2.0 * A * omega * np.cos(2.0 * omega * t_grid)
    psi_grid = np.unwrap(np.arctan2(dY_grid, dX_grid))
    psi = float(psi_grid[-1])

    return np.array([X, Y, psi])


def lane_change(t, v_long=1.0, lane_width=0.4, t0=12.0, k=2.0):
    """
    t: simulation time.

    v_long: Longitudinal velocity of the vehicle.

    lane_width: Lateral displacement in meters.

    t0 : Time at which the lane change is halfway completed.

    k :
    Smaller k: Faster, sharper lane change.
    Larger k: Slower, smoother lane change.
    """
   

    X = v_long * t

    Y = lane_width * sig((t-t0) / k)

    eps = 1e-3

    X2 = v_long * (t + eps)
    Y2 = lane_width * sig((t-t0 + eps) / k)

    psi = np.arctan2(Y2 - Y, X2 - X)

    return np.array([X, Y, psi])

def double_lane_change(t, v_long=1.0, lane_width=2, t0=9.0, k=1.0, dwell_time=15.0):
    """
    dwell_time: Time spent in the adjacent lane before returning.
    """

    X = v_long * t

    # First lane change
    sig1 = sig((t - t0) / k)

    # Return lane change
    sig2 = sig((t - (t0 + dwell_time)) / k)

    Y = lane_width * (sig1 - sig2)

    eps = 1e-3

    X2 = v_long * (t + eps)

    sig1_2 = 1.0 / (1.0 + np.exp(-(t + eps - t0) / k))
    sig2_2 = 1.0 / (1.0 + np.exp(-(t + eps - (t0 + dwell_time)) / k))

    Y2 = lane_width * (sig1_2 - sig2_2)

    psi = np.arctan2(Y2 - Y, X2 - X)

    return np.array([X, Y, psi])