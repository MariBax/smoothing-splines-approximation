import numpy as np

class OriginalFunction():
    """Original function f as a Fourier basis with random coefficients
   
    Parameters
    ----------
    n : int, should be even as it is Fourier basis
        The number of basis functions.
        
    c : np.array
        Random coefficients.

    """
    
    
    def __init__(self, n, seed=20):
        self.n = n
        self.c = self._construct_coeffs(seed)
        
    def __call__(self, x):
        """Compute f(x)
        x : np.array
        """
        basis_x = self._basis(x)
        return np.dot(basis_x.T, self.c)
    
    def derivative(self, x):
        """Compute f'(x)
        x : np.array
        """
        dbasis_x = self._dbasis(x)
        return np.dot(dbasis_x.T, self.c)        

    def _construct_coeffs(self, seed):
        np.random.seed(seed)
        c = np.zeros(self.n)
        c[:10] = np.random.normal(loc=0, scale=1, size=10) # note that indexation from 0
        c[10:] = np.random.normal(loc=0, scale=1, size=self.n - 10) / np.square(np.arange(11, self.n + 1) - 10)
        return c

    def _basis(self, x):
        basis_x = np.zeros((self.n, x.shape[0]))
        ind_range = np.arange(0, self.n / 2).reshape(-1, 1)
        basis_x[1::2, :] = np.sin(2 * np.pi * x * ind_range) # odd rows
        basis_x[::2, :] = np.cos(2 * np.pi * x * ind_range) # even rows
        return basis_x

    def _dbasis(self, x):
        dbasis_x = np.zeros((self.n, x.shape[0]))
        ind_range = np.arange(0, self.n / 2).reshape(-1, 1)
        dbasis_x[1::2, :] = 2 * np.pi * ind_range * np.cos(2 * np.pi * x * ind_range) # odd rows
        dbasis_x[::2, :] = - 2 * np.pi * ind_range * np.sin(2 * np.pi * x * ind_range) # even rows        
        return dbasis_x