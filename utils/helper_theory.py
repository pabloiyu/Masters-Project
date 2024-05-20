from scipy.integrate import quad, nquad
import numpy as np
from numpy.linalg import det, inv
import sympy as sp
from sympy.utilities.lambdify import lambdify
import torch

class DerivativesActivationFunction():
    """
    This class is used to calculate the derivatives of the activation function. Given an
    activation function f(x) it uses sympy to analytically calculate the necessary derivatives.
    It then uses lambdify to convert these sympy expressions into numpy callable functions.
    """
    def __init__(self, use_identity_activation=False, phi1=1):
        phi1 = 1
        x = sp.symbols('x')

        if use_identity_activation:
            f = phi1*x
        else:
            f = sp.tanh(x)
            
        f_d1 = sp.diff(f, x, 1)
        f_d2 = sp.diff(f, x, 2)
        f_d3 = sp.diff(f, x, 3)
        f_d4 = sp.diff(f, x, 4)
        fsq_d2 = sp.diff(f**2, x, 2)
        fsq_d4 = sp.diff(f**2, x, 4)

        self.f = lambdify(x, f, "numpy")
        self.f_d1 = lambdify(x, f_d1, "numpy")
        self.f_d2 = lambdify(x, f_d2, "numpy")
        self.f_d3 = lambdify(x, f_d3, "numpy")
        self.f_d4 = lambdify(x, f_d4, "numpy")
        self.fsq_d2 = lambdify(x, fsq_d2, "numpy")
        self.fsq_d4 = lambdify(x, fsq_d4, "numpy")

def gaussian_pdf(mu, sigma, x):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu)**2) / (2 * sigma**2))

def expectation_gaussian_measure(func, mu, sigma):
    """
    This function calculates the expectation of a function with respect to a Gaussian measure.
    """
    # We can ignore error in theoretical values as this will be due to numerical precision of doubles.
    gaussian = lambda x: gaussian_pdf(mu, sigma, x)   #partial(gaussian_pdf, mu, sigma)
    result, _ = quad(lambda x: func(x) * gaussian(x), -np.inf, np.inf)
    
    return result

def gaussian_pdf_ndim(x, cov, det_cov, inv_cov, mu=0):
    """
    This function calculates the n-dimensional Gaussian PDF.
    """
    # Define a normalized n-dimensional Gaussian
    # with mean mu and covariance matrix cov
    d = len(x)
    return 1 / ((2*np.pi)**(d/2) * np.sqrt(det_cov)) * np.exp(-0.5 * (x-mu) @ inv_cov @ (x-mu))

def expectation_gaussian_measure_ndim(func1, func2, first_var, second_var, cov, mu=0, simplify=True):
    """
    This function calculates the expectation of a function with respect to an n-dimensional Gaussian measure.
    """
    # In the 1-dimensional case, we revert to the simpler function
    # *** This can happen when either we are considering a diagonal term and two variables are the same
    # *** Or if someone has given us a covariance matrix where every element is equal. In this case
    # I assume that the person wants to integrate over the same variable.
    if first_var == second_var or cov[first_var, second_var] == cov[first_var, first_var]:
        return expectation_gaussian_measure(lambda x: func1(x)*func1(x), mu, np.sqrt(cov[first_var, first_var]))
    
    # As our expectation will only be with 2 of the n variables, we can integrate/marginalise out 
    # the rest of the n variables. 
    if simplify:
        cov = np.array([[cov[first_var, first_var],  cov[first_var, second_var]], 
                        [cov[second_var, first_var], cov[second_var, second_var]]]) 
        first_var = 0
        second_var = 1   
    
    n = cov.shape[0]
    det_cov = det(cov)
    inv_cov = inv(cov)
    
    gaussian_ndim = lambda x: gaussian_pdf_ndim(x, cov, det_cov, inv_cov, mu)
    
    # Integration only occurs over two arguments of the n-dimensional Gaussian
    simplified_func = lambda x: func1(x[first_var]) * func2(x[second_var])
    
    # Define the integrand as the product of the function and the Gaussian PDF
    integrand = lambda *x: simplified_func(x) * gaussian_ndim(np.array(x))
    
    # Define the integration limits
    x_limits = [[-np.inf, np.inf] for _ in range(n)]
    
    # Perform the integration
    result, _ = nquad(integrand, x_limits)

    return result

def compute_KV_1input(x, num_inputs, width_hidden_layer, num_layers, Cb, Cw, use_identity_activation=False):
    """
    This function calculates the K and V values for a network with 1 input. It calculates both the leading
    order term and the O(1/n) corrections.
    """
    list_width_network = [num_inputs] + [width_hidden_layer] * (num_layers-1) + [1]

    k0_1 = Cb + Cw * np.mean(np.square(x))
    k1_1 = 0
    V_1 = 0

    list_k0 = np.zeros((num_layers))
    list_k1 = np.zeros((num_layers))
    list_V = np.zeros((num_layers))
    list_k0[0] = k0_1
    list_k1[0] = k1_1
    list_V[0] = V_1
    
    deriv_handler = DerivativesActivationFunction(use_identity_activation)

    for lp1 in range(2,num_layers+1):
        k0_l = list_k0[lp1-2]
        k1_l = list_k1[lp1-2]
        V_l = list_V[lp1-2]
        
        # Let's first calculate k0 for the lth layer, given the l-1 layer values
        k0_lp1 = Cb + Cw*expectation_gaussian_measure(lambda y: deriv_handler.f(y)**2, 0, np.sqrt(k0_l))
        list_k0[lp1-1] = k0_lp1
        
        # Let's now calculate k1 for the lth layer
        chi_parallel_k0_l = Cw / 2 * expectation_gaussian_measure(deriv_handler.fsq_d2, 0, np.sqrt(k0_l))
        k1_lp1 = (list_width_network[lp1-1]/list_width_network[lp1-2]) * (chi_parallel_k0_l * k1_l + Cw / 8 * V_l * expectation_gaussian_measure(deriv_handler.fsq_d4, 0, np.sqrt(k0_l)))
        list_k1[lp1-1] = k1_lp1
        
        # Let's now calculate V_lplus1
        diff_expectation_gaussian_measure = expectation_gaussian_measure(lambda y: deriv_handler.f(y)**4, 0, np.sqrt(k0_l)) - expectation_gaussian_measure(lambda y: deriv_handler.f(y)**2, 0, np.sqrt(k0_l))**2
        V_lp1 = (list_width_network[lp1-1]/list_width_network[lp1-2]) * chi_parallel_k0_l**2 * V_l + Cw**2 * diff_expectation_gaussian_measure
        list_V[lp1-1] = V_lp1
        
    return list_k0, list_k1, list_V

def compute_theoretical_prediction_NTK_initialisation(X, num_inputs, width_hidden_layer, num_layers, lambda_w_inputs, lambda_w_hidden_layer, lambda_b, Cb, Cw,  use_identity_activation=False):
    num_data_points = X.shape[0]
    
    dotted_inputs = np.einsum('im,km -> ik', X, X)
    deriv_handler = DerivativesActivationFunction(use_identity_activation)

    # X is of size (num_data_points, 384)
    K_i = Cb + Cw/num_inputs * dotted_inputs
    theta_i = lambda_b + lambda_w_inputs * dotted_inputs  #lambda_b + lambda_w/num_inputs * dotted_inputs 
    
    for _ in range(int(num_layers)-1):
        if use_identity_activation:
            theta_i = lambda_b + lambda_w_hidden_layer * K_i * width_hidden_layer + Cw * theta_i  # lambda_b + lambda_w * K_1 + Cw * theta_1 

        else:
            expectation_func = np.zeros_like(theta_i)
            expectation_deriv = np.zeros_like(theta_i)

            for i in range(num_data_points):
                for j in range(num_data_points):
                    expectation_func[i,j]  = expectation_gaussian_measure_ndim(deriv_handler.f, deriv_handler.f, i, j, K_i)
                    expectation_deriv[i,j] = expectation_gaussian_measure_ndim(deriv_handler.f_d1, deriv_handler.f_d1, i, j, K_i)
            
            theta_i = lambda_b + lambda_w_hidden_layer * expectation_func * width_hidden_layer + Cw * expectation_deriv * theta_i  # lambda_b + lambda_w * expectation_func + Cw * expectation_deriv * theta_1
            
        old_K_i = np.copy(K_i)
        for i in range(num_data_points):
            for j in range(num_data_points):
                K_i[i,j] = Cb + Cw * expectation_gaussian_measure_ndim(deriv_handler.f, deriv_handler.f, i, j, old_K_i)
            
    return theta_i

def frobenius_norm(A):
    return torch.sqrt(torch.sum(A**2))

def get_critical_learning_rate(H):
    """
    Given the Neural Tangent Kernel, calculate its min and max eigenvalues.
    """
    eigvals, _ = torch.linalg.eig(H)  # Get eigenvalues, ignore eigenvectors 
    min_eigval = torch.min(eigvals.real)
    max_eigval = torch.max(eigvals.real)
    
    if torch.any(eigvals.imag != 0):
        raise ValueError("Eigenvalues are not real!")

    critical_learning_rate = 2 / (min_eigval + max_eigval)

    return critical_learning_rate
    
def probability_distribution_phi(x, K, V):
    return np.exp(-x**2/(2*K) - V*x**4/8)

def main():
    # Let's define a simple function and check if the expectation_gaussian_measure works
    func = lambda x: x**2
    mu = 0
    sigma = 10
    result = expectation_gaussian_measure(func, mu, sigma)
    
    print("1 dimensional Gaussian", result)
    
    # Define your covariance matrix
    cov = np.eye(3)  # Identity matrix for simplicity

    # Both results should be the same
    deriv_handler = DerivativesActivationFunction(use_identity_activation=False)
    result_simple = expectation_gaussian_measure_ndim(deriv_handler.f_d1, deriv_handler.f_d1, 0, 1, cov)
    result_long   = expectation_gaussian_measure_ndim(deriv_handler.f_d1, deriv_handler.f_d1, 0, 1, cov, simplify=False)

    print("Integration long:", result_long)
    print("Integration simple:", result_simple)

if __name__ == "__main__":
    main()