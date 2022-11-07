from numpy import array, pi, diag, zeros, sum, sin, add, subtract, multiply
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
        self.finite = self.finite(self.N, self.c)
        self.ring = self.ring(self.N)

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
        A = rand( N, N)
        A_symm = (A + A.T)/2
        A_symm = (A_symm < c/(N-1)).astype(int)
        return A_symm - diag(diag(A_symm))

    def force(self, spins: array) -> array:
        """
        Computes the force vector for the current state of the system

        Returns:
            array: _description_
        """
        s_outer = subtract.outer(spins, spins)
        s_ring = s_outer * self.ring
        s_finite = s_outer * self.finite
        f = add(self.J0 * sum(sin(s_ring), axis=0), self.J * sum(sin(s_finite), axis=0))
        return f
