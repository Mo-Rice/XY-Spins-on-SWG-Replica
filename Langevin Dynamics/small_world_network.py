from numpy import array, pi, diag, zeros, sum, sin, subtract
from numpy.random import rand, randn


class SmallWorldNetwork():

    def __init__(self, N: int, c: float, J: float, J0: int = 1):
        """
        Initializes a small world network class for use in a Langevin
        simulation

        Args:
            N (int): Number of XY spins in the network
            c (float): Connectivity of the underlying finite graph
        """
        self.N = N
        self.ring = self.ring(self.N)
        self.finite = self.finite(self.N)
        self.spins = randn(N)*(2*pi)
        self.J0 = J0
        self.J = J
        self.c = c

    def ring(N: int) -> array:
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

    def finite(N: int, c: float) -> array:
        """
       Generates the adjacency matrix of the finite connectivity graph

        Args:
            N (int): Number of XY spins in the network
            c (float): Average connectivity

        Returns:
            array: Finite connectivity graph adjacency matrix
        """
        A = rand(N, N)
        A_symm = (A + A.T)/2
        A_symm = (A_symm < c/(N-1)).astype(int)
        return A_symm - diag(diag(A_symm))

    def force(self) -> array:
        """
        Computes the force vector for the current state of the system

        Returns:
            array: _description_
        """
        ss = sin(subtract.outer(self.spins, self.spins))
        return (self.J0*sum(self.ring*ss, axis=1) +
                self.J*sum(self.finite*ss, axis=1))
