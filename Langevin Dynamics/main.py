import numpy as np
from langevin_dynamics import langevinDynamics
from small_world_network import smallWorldNetwork
from tqdm import tqdm
from os.path import join

# network parameters
N = 10
c_s = np.arange(0.1, 1.1, 0.1)
J = 0
J0 = 2

# simulation parameters
T = 1
n = int(np.sqrt(N))
dt = 0.01
b_n = 10

path = "../../data"
stats = np.zeros((4, len(c_s)))

#for i in tqdm(range(len(c_s)), desc="Sweeping connectivity"):
SWN = smallWorldNetwork(N, (1/10), J, J0)
LD = langevinDynamics(SWN, n, dt, T, b_n)
LD.passage_time = 100
LD.dynamics()
#langevinDynamics.save(filename)
_, _, m, _, _, q, c_x, c_y = LD.get_statistics()
#stats[:, i] = np.array([m, q, c_x, c_y])

print('-'*20)
print(langevinDynamics.current_state)
print(langevinDynamics.samples)