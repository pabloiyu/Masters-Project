from scipy.integrate import quad, nquad
import numpy as np
from numpy.linalg import det, inv
import sympy as sp
from sympy.utilities.lambdify import lambdify
from utils.helper_computation import use_identity_activation

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

f = lambdify(x, f, "numpy")
f_d1 = lambdify(x, f_d1, "numpy")
f_d2 = lambdify(x, f_d2, "numpy")
f_d3 = lambdify(x, f_d3, "numpy")
f_d4 = lambdify(x, f_d4, "numpy")
fsq_d2 = lambdify(x, fsq_d2, "numpy")
fsq_d4 = lambdify(x, fsq_d4, "numpy")


def gaussian_pdf(mu, sigma, x):
    return 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-((x - mu)**2) / (2 * sigma**2))

def expectation_gaussian_measure(func, mu, sigma):
    # We can ignore error in theoretical values as this will be due to numerical precision of doubles.
    gaussian = lambda x: gaussian_pdf(mu, sigma, x)   #partial(gaussian_pdf, mu, sigma)
    result, _ = quad(lambda x: func(x) * gaussian(x), -np.inf, np.inf)
    
    return result

def gaussian_pdf_ndim(x, cov, det_cov, inv_cov, mu=0):
    # Define a normalized n-dimensional Gaussian
    # with mean mu and covariance matrix cov
    d = len(x)
    return 1 / ((2*np.pi)**(d/2) * np.sqrt(det_cov)) * np.exp(-0.5 * (x-mu) @ inv_cov @ (x-mu))

def expectation_gaussian_measure_ndim(func1, func2, first_var, second_var, cov, mu=0, simplify=True):
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
    result_simple = expectation_gaussian_measure_ndim(f_d1, f_d1, 0, 1, cov)
    result_long   = expectation_gaussian_measure_ndim(f_d1, f_d1, 0, 1, cov, simplify=False)

    print("Integration long:", result_long)
    print("Integration simple:", result_simple)

if __name__ == "__main__":
    main()