import numpy as np

def figure_eight(t, A= 5.0, omega= 0.2):
    '''
    discrete points figure 8
    Return:  [X, Y, psi]
    '''

    dt_small = 1e-3

    X = A * np.sin(omega * t)
    Y = (A * np.sin(2 * omega * t))

    X2 = A * np.sin(omega * (t + dt_small))
    Y2 = (A * np.sin(2 * omega * (t + dt_small)))

    psi = np.arctan2(Y2 - Y, X2 - X)

    return np.array([X, Y, psi])

def double_lane_change(t, v_long= 2.0):
    '''
    Vehicel moves along +X.
    Y is a pulse.
    '''

    def sig(x):
        return 1.0 / (1.0 + np.exp(-x))
    
    X = v_long * t
    Y = 1.5 * (sig(X-5) - sig(X-20) - sig(X-35) + sig(X-50))
    eps = 1e-3

    X2 = X + eps
    Y2 = 1.5 * (sig(X2-5) - sig(X2-20) - sig(X2-35) + sig(X2-50))
    psi = np.arctan2(Y2 - Y, eps)
    
    return np.array([X, Y, psi])