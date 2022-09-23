from tkinter import W
from numba import njit
import numpy as np
from small_world_network import smallWorldNetwork


class langevinDynamics():
    def __init__(self, SWN: smallWorldNetwork, n: int, nb: int,
                 dt: float, T: float):
        self.SWN = SWN
        self.N = SWN.N
        self.n = n
        self.nb = nb
        self.dt = dt
        self.T = T
        self.samples = np.zeros(self.N, n)
        self.current_state = np.zeros(self.N)
        
        