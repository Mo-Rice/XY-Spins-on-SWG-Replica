import numpy as np
import matplotlib.pyplot as plt
from langevin_dynamics import langevinDynamics
from small_world_network import smallWorldNetwork

# network parameters
N = 1000
c = (1/0.6)
J = 1
J0 = -0.5

# simulation parameters
T = 0.1
n = int(np.sqrt(N))
dt = 0.1
b_n = 1000

filename = "../../data/test.csv"
SWN = smallWorldNetwork(N, c, J, J0)
langevinDynamics = langevinDynamics(SWN, n, dt, T, b_n)
langevinDynamics.passage_time = 100
langevinDynamics.dynamics()
s = langevinDynamics.samples
langevinDynamics.save(filename)
x = np.cos(s)
y = np.sin(s)
m_x = x.mean()
m_y = y.mean()
m = np.sqrt(m_x**2 + m_y**2)
q_xx = np.mean(np.power(x.mean(axis=1),2))
q_yy = np.mean(np.power(y.mean(axis=1),2))
q = q_xx + q_yy
print('-'*20)
print(f'm_x: {m_x} \n m_y: {m_y} \n m: {m} \n q_xx: {q_xx} \n q_yy: {q_yy} \n q: {q}')
print(langevinDynamics.current_state)
print(langevinDynamics.samples)