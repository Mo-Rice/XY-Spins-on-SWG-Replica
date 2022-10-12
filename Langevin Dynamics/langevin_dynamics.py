from numba import njit
import numpy as np
from numba import jit
from small_world_network import smallWorldNetwork


class langevinDynamics():
    def __init__(self, SWN: smallWorldNetwork, n: int,
                 dt: float, T: float):
        """

        :param SWN: generated small world network object
        :param n: number of runs
        :param nb: number of desired burn in steps (will be chosen to be >= sqrt(N)
        :param dt: evolution time step
        :param T: temperature
        """
        self.SWN = SWN
        self.N = SWN.N
        self.n = n
        self.dt = dt
        self.T = T
        self.samples = np.zeros((self.N, n))
        self.current_state = SWN.spins # randomly initialized spin vector from SWN

    def state_update(self, update: np.array):
        self.current_state = update
    def dynamics(self) -> np.array:
        system_fluc = int(np.sqrt(self.N))
        sigma = np.sqrt(2*self.T/self.dt)

        for i in range(self.n):
            for ii in range(system_fluc):
                ds = self.SWN.force()
                s_i = self.current_state.copy() - ds*self.dt + sigma*np.random.randn(self.N)*self.dt
                self.state_update(s_i)

            self.samples[:, i] = self.current_state