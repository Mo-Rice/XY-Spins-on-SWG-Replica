from numba import njit
import numpy as np
from numba import jit
from numpy.random import default_rng
from small_world_network import smallWorldNetwork
from tqdm import tqdm
import pandas as pd


class langevinDynamics():
    def __init__(self, SWN: smallWorldNetwork, n: int,
                 dt: float, T: float, b_n: int):
        """

        :param SWN: generated small world network object
        :param n: number of runs
        :param nb: number of desired burn in steps (will be chosen to be >= sqrt(N)
        :param dt: evolution time step
        :param T: temperature
        :param b_n: initial burn in period
        """
        self.SWN = SWN
        self.N = SWN.N
        self.n = n
        self.dt = dt
        self.T = T
        self.b_n = b_n
        self.rng = default_rng()
        self.sigma = np.sqrt(2 * self.T * self.dt)
        self.samples = np.zeros((self.N, n))
        self.passage_time = int(self.N / self.dt)
        self.current_state = SWN.spins  # randomly initialized spin vector from SWN

    def state_update(self, update: np.array):
        self.current_state = update

    def burn_in(self):
        for i in tqdm(range(self.b_n), desc="Burning in the system"):
            ds = np.add(self.SWN.force(self.current_state) * self.dt,
                        self.rng.normal(size=self.N) * np.sqrt(2 * self.T * self.dt))
            s_i = np.add(self.current_state, ds)
            self.state_update(s_i)

    def dynamics(self):
        self.burn_in()

        for i in tqdm(range(self.n), desc="Sampling"):
            for ii in range(self.passage_time):
                ds = np.add(self.SWN.force(self.current_state) * self.dt,
                            self.rng.normal(size=self.N) * np.sqrt(2 * self.T * self.dt))
                s_i = np.add(self.current_state, ds)
                self.state_update(s_i)

            self.samples[:, i] = self.current_state

    def save(self, filename: str, header=True) -> None:
        df = pd.DataFrame(self.samples)
        if header:
            with open(filename, 'w') as file:
                file.writelines(
                    f'Simulation settings: \n N = {self.N} \n c = {self.SWN.c} \n n = {self.n} \n T = {self.T} \n dt = {self.dt} \n b = {self.b_n}\n')

        df.to_csv(filename, mode="a", header=False, index=False)

    def get_statistics(self):
        x = np.cos(self.samples)
        y = np.sin(self.samples)
        m_x = x.mean()
        m_y = y.mean()
        m = np.sqrt(m_x ** 2 + m_y ** 2)
        q_xx = np.mean(np.power(x.mean(axis=1), 2))
        q_yy = np.mean(np.power(y.mean(axis=1), 2))
        q = q_xx + q_yy
        c_x = np.mean(np.multiply(x, np.roll(x, 1)))
        c_y = np.mean(np.multiply(y, np.roll(y, 1)))

        return m_x, m_y, m, q_xx, q_yy, q, c_x, c_y
