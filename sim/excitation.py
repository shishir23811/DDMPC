import numpy as np

def generate_pe_input(N, dt, seed= 0):
    '''
    N: Horizon
    '''

    rng = np.random.default_rng(seed)
    t = np.arange(N) * dt

    delta_amp = np.deg2rad(12.0)
    freqs_steer = [0.15, 0.3, 0.45, 0.6, 0.75, 1.0]
    n_freqs_steer = len(freqs_steer)

    delta = sum(
        (delta_amp/n_freqs_steer) * 
        np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        for f in freqs_steer
    )
    delta += rng.normal(0.0, np.deg2rad(0.5), N)

    a_bias = 0.3
    a_amp = 0.4
    freqs_acc = [0.05, 0.11, 0.19]
    n_freqs_acc = len(freqs_acc)

    a = a_bias + sum(
        (a_amp/ n_freqs_acc) *
        np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
        for f in freqs_acc
    )

    return np.stack([delta, a], axis= 1)