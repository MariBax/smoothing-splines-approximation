import numpy as np
import scipy.integrate as integrate

class SmoothingSpline():
    """Smoothing spline
   
    Parameters
    ----------
    knots : ordered np.array
        Ordered equidistant points that divide the interval [0,1].
    
    x : np.array
        Design points.
        
    y : np.array
        Observed values (f(x) + noise), where f is original function.
    
    alpha: float, default: 1

    """
    
    def __init__(self, knots, x, y, alpha=1):        
        self.x_min = knots[0]
        self.x_max = knots[-1]
        self.knots = knots
        self.knots_size = len(knots)
        self.basis_size = self.knots_size + 3
        self.alpha = alpha
        self.y = y
        
        self.basis_x = self._basis(x) 
        self.A_integral = self._compute_A_integral()
        self.S = self._compute_S()
        self.theta = self._compute_optimal_theta()
    
    def __call__(self, x0):
        basis_x0 = self._basis(x0)
        return np.dot(basis_x0.T, self.theta)
    
    def derivative(self, x0):
        dbasis_x0 = self._dbasis(x0)
        return np.dot(dbasis_x0.T, self.theta)
    
    def compute_K(self, x0):
        dbasis_x0 = self._dbasis(x0)
        K = np.dot(dbasis_x0.T, self.S)
        return K
        
    def _basis(self, x):
        basis = np.zeros((self.basis_size, x.shape[0]))
        for row_id in range(4):
            basis[row_id, :] = x**row_id      
        for row_id in range(4, self.basis_size):
            zero = np.zeros(x.shape[0])
            basis[row_id, :] = np.maximum(zero, (x - self.knots[row_id - 4])**3)    
        return basis

    def _dbasis(self, x):
        dbasis = np.zeros((self.basis_size, x.shape[0]))
        dbasis[0, :] = np.zeros(x.shape[0])
        dbasis[1, :] = np.ones(x.shape[0])
        dbasis[2, :] = 2 * x
        dbasis[3, :] = 3 * x**2
        for row_id in range(4, self.basis_size):
            zero = np.zeros(x.shape[0])
            dbasis[row_id, :] = 3 * np.maximum(zero, x - self.knots[row_id - 4])**2            
        return dbasis
    
    def _d2_basis_pattern_1(self, i):
        if i == 0:
            return lambda x: 0
        elif i == 1:
            return lambda x: 0
        elif i == 2:
            return lambda x: 2
        elif i == 3:
            return lambda x: 6 * x

    def _d2_basis_pattern_2(self, x0):
        return lambda x: 6 * (x - x0)   

    def _compute_A_integral(self):
        d2_basis_1 = [self._d2_basis_pattern_1(i) for i in range(4)]
        d2_basis_2 = [self._d2_basis_pattern_2(self.knots[i]) for i in range(0, self.knots_size - 1)]
        d2_basis_func = d2_basis_1 + d2_basis_2
        
        A_integral = np.zeros((self.basis_size, self.basis_size))

        for row_id in range(A_integral.shape[0]):
            for col_id in range(A_integral.shape[0]):
                integrand = lambda x: d2_basis_func[row_id](x) * d2_basis_func[col_id](x)
                if row_id >= 4 or col_id >= 4:
                    lower_limit = self.knots[max(row_id, col_id) - 4]            
                    A_integral[row_id, col_id] = integrate.quad(integrand, 
                                                                lower_limit, self.x_max)[0]
                else:
                    A_integral[row_id, col_id] = integrate.quad(integrand, 
                                                                self.x_min, self.x_max)[0]
        return A_integral    
    
    def _compute_S(self):
        inner_sum = self.basis_x @ self.basis_x.T + self.alpha * self.A_integral
        S = np.linalg.pinv(inner_sum) @ self.basis_x 
        return S
    
    def _compute_optimal_theta(self):
        theta = np.dot(self.S, self.y)
        return theta