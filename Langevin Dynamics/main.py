import numpy as np
import matplotlib.pyplot as plt
from langevin_dynamics import langevinDynamics
from small_world_network import smallWorldNetwork

# network parameters
N = 100
c = 0.1
J = 1
J0 = 1

# simulation parameters
T = 0.3
n = 10
dt = 0.01

SWN = smallWorldNetwork(N, c, J, J0)
langevinDynamics = langevinDynamics(SWN, n, dt, T)
langevinDynamics.dynamics()
print(langevinDynamics.current_state)
print(langevinDynamics.samples)