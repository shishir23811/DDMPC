import numpy as np

def hankel(signal, L):
    N, d = signal.shape
    cols = N - L + 1
    H = np.zeros((L * d, cols))

    for i in range(L):
        H[i*d: (i+1)*d, :] = signal[i:i+cols].T

    return H

def is_pe(U, L):
    Hu = hankel(U, L)
    rank = np.linalg.matrix_rank(Hu)
    expected = L * U.shape[1]
    status = 'PE' if rank == expected else 'Not PE'
    
    print(f'Hankel order: {L}, Rank= {rank}, Expected rank= {expected}')

    return rank == expected