import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from numpy import zeros, sum, sqrt
from numpy.random import normal
from tqdm import tqdm


# Helper Functions
def finite(N: int, c: float) -> np.ndarray:
    """
    Generates the adjacency matrix of the finite connectivity graph

    Args:
        N (int): Number of XY spins in the network
        c (float): Average connectivity

    Returns:
        array: Finite connectivity graph adjacency matrix
    """
    A = np.random.uniform(0, 1, size=(N, N))
    A_mask = np.triu((A < c/N).astype(int), k=1)
    A_symm = (A_mask + A_mask.T) 
    return A_symm


def ring(N: int) -> np.ndarray:
    """
    Generates the adjacency matrix of the ring

    Args:
        N (int): Number of XY spins in the network

    Returns:
        array: Ring adjacency matrix
    """
    A = np.zeros((N, N))
    pos = 0

    for i in range(N):
        A[i][(pos+1) % N] = 1
        A[i][(pos-1) % N] = 1
        pos += 1
    return A.astype(int)

def get_statistics(samples: np.ndarray):
    x = np.cos(samples)
    y = np.sin(samples)
    m_x = x.mean()
    m_y = y.mean()
    m = np.sqrt(m_x ** 2 + m_y ** 2)
    q_xx = np.mean(np.power(x.mean(axis=1), 2))
    q_yy = np.mean(np.power(y.mean(axis=1), 2))
    q = q_xx + q_yy
    c_x = np.mean(np.multiply(x, np.roll(x, 1)))
    c_y = np.mean(np.multiply(y, np.roll(y, 1)))

    return m_x, m_y, m, q_xx, q_yy, q, c_x, c_y

@nb.njit(parallel=True, cache=True)
def force(spins: np.ndarray, A_r: np.ndarray, A_f: np.ndarray, J: float, J0: float) -> np.ndarray:
    t = zeros((N,N))
    for i in range(N):
        for j in range(N):
            t[i, j] = spins[i] - spins[j]
    t = np.sin(t)
    return J*sum(t*A_f, axis=1) + J0*sum(t*A_r, axis=1)

def loop(s: np.ndarray, n: int, A_r: np.ndarray, A_f: np.ndarray, J: float, J0: float, T:float, dt: float) -> np.ndarray:
    for i in range(n):
        s += (normal(size=s.size)* sqrt(2 * T * dt) - force(s, A_r, A_f, J, J0)*dt)
    return s
    

# Network Parameters
N = 1000            # Numbers of Spings
c = 1               # Finite Connectivity
J = 1.0             # Finite Graph Strength
J0 = 0.5            # Ring Strength

# Simulation Parameters
T = 0.1               # Temperature
dt = 0.1           # Time Step
n = 40#int(np.sqrt(N)) # Number of Runs (computed as sqrt(N) for proper stats)
n_s = 100#int((N/dt))   # Number of Steps Per Run (computed to allow enough time for system to evolve)
b_n = 1000          # Burn-in Time

# Simulation Containers
s_0 = np.random.uniform(-1,1, N)*np.pi # Initial State
s_n = np.zeros((n+1, N))            # Samples
A_r = ring(N)
A_f = finite(N, c)

print('Burning in the system...\n')
s_n[0, :] = loop(s_0, b_n, A_r, A_f, J, J0, T, dt)
print('Done!\n')

for i in tqdm(range(n), desc="Sampling"):
    s_n[i+1, :] = loop(s_n[i, :], n_s, A_r, A_f, J, J0, T, dt) 

m_x, m_y, m, q_xx, q_yy, q, c_x, c_y = get_statistics(s_n)

print('blah')