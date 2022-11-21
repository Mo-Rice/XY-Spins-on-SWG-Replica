from numpy import array, pi, diag, zeros, sum, sin, add, subtract, multiply, triu
from numpy.random import rand, uniform



class smallWorldNetwork():
    def __init__(self, N: int, c: float, J: float, J0: float = 1.0):
        """
        Initializes a small world network class for use in a Langevin
        simulation

        Args:
            N (int): Number of XY spins in the network
            c (float): Connectivity of the underlying finite graph
        """
        self.N = N
        self.J0 = J0
        self.J = J
        self.c = c
        self.spins = uniform(-1,1,self.N)*pi
        self.A_f = self.finite(self.N, self.c)
        self.A_r = self.ring(self.N)

    def ring(self, N: int) -> array:
        """
        Generates the adjacency matrix of the ring

        Args:
            N (int): Number of XY spins in the network

        Returns:
            array: Ring adjacency matrix
        """
        A = zeros((N, N))
        pos = 0

        for i in range(N):
            A[i][(pos+1) % N] = 1
            A[i][(pos-1) % N] = 1
            pos += 1
        return A.astype(int)

    def finite(self, N: int, c: float) -> array:
        """
       Generates the adjacency matrix of the finite connectivity graph

        Args:
            N (int): Number of XY spins in the network
            c (float): Average connectivity

        Returns:
            array: Finite connectivity graph adjacency matrix
        """
        A = uniform( 0, 1, size=(N, N))
        A_mask = triu((A < c/N).astype(int), k=1)
        A_symm = (A_mask + A_mask.T) / 2
        return A_symm

    def force(self, spins: array) -> array:
        """
        Computes the force vector for the current state of the system

        Returns:
            array: _description_
        """
        s_outer = subtract.outer(spins, spins)
        s_ring = s_outer * self.A_r
        s_finite = s_outer * self.A_f
        f = add(self.J0 * sum(sin(s_ring), axis=0), self.J * sum(sin(s_finite), axis=0))
        return f
