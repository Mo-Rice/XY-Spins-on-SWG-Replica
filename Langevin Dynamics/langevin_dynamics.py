from numba import njit
import numpy as np
from small_world_network import SmallWorldNetwork


class langevinDynamics():
    def __init__(self, SWN: SmallWorldNetwork, n: int, nb: int,
                 dt: float, T: float):
        self.SWN = SWN
        self.N = SWN.N
        self.n = n
        self.nb = nb
        self.dt = dt
        self.T = T
        self.samples = np.zeros(self.N, n)
        self.current_state = np.zeros(self.N)

    def dynamics(self) -> np.array:
        system_fluc = np.sqrt(self.N)

        
        