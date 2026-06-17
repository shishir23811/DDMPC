import numpy as np
import os
from sim.bicycle import KinematicBicycle
from sim.excitation import generate_pe_input
from sim.check_pe import is_pe

def collect(N= 1500, dt= 0.05, seed= 0, path= 'data/open_loop.npz'):
    np.random.seed(seed)
    veh = KinematicBicycle(dt= dt, noise_std= {'X': 0.02, 'Y': 0.03, 'psi': 0.01})
    
    U = generate_pe_input(N, dt, seed= seed)
    Y = np.zeros((N, 3))

    Y[0] = veh.reset(x0= [0.0, 0.0, 0.0, 1.0])

    for k in range(N-1):
        Y[k+1] = veh.step(U[k])

    U = U[:-1]
    Y = Y[1:]

    is_satis = is_pe(U, L= 36)
    print(is_satis)
    if(is_satis):
        os.makedirs(os.path.dirname(path), exist_ok= True)
        np.savez(path, U= U, Y= Y)
    else:
        print("Input is not persistently exciting.")
    
    return U, Y

if __name__ == '__main__':
    collect()