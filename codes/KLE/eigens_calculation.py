# Libraries
import numpy as np
from scipy.optimize import brentq

# Parameters & save path
L = 100
eta = 50
variance = 3
nkl = 9
save_path_1 = '.../Geo-PINN/data/eigens/' # replace with the correct path
save_path_2 = 'eta=' + str(eta) + '_var=' + str(variance) + '/'
save_path = save_path_1 + save_path_2

# Finding positive roots
# equation definition
def equation(w):
    return (eta**2 * w**2 - 1) * np.sin(w * L) - 2 * eta * w * np.cos(w * L)

# searching range and step
w_min, w_max, step = 0, 10, 0.00001

# initialize
w_values = np.arange(w_min, w_max, step)
roots = []

# function to refine the root estimation using Brent's method
def find_root(interval):
    try:
        root = brentq(equation, interval[0], interval[1])
        return root
    except ValueError:
        return None

# searching for roots
for i in range(len(w_values) - 1):
    w1, w2 = w_values[i], w_values[i + 1]
    if equation(w1) * equation(w2) < 0:  # Sign change indicates a root
        # Refine root estimation in this interval
        root = find_root([w1, w2])
        if root is not None and root not in roots:
            roots.append(root)

# selecting the first 9 positive roots
roots = np.array(sorted(roots)[:nkl])
np.save(save_path + 'selected_wn', roots)

# Definition of the functions to calculate eigenvalues and eigenfunctions
x_values = np.linspace(0, -99, num=100)

def eigenvalue(eta, variance, wn):
    """
    Calculate the eigenvalue λ_n.

    Parameters:
    eta (float): The parameter eta.
    sigma_Y (float): The parameter sigma_Y.
    wn (float): The eigenvalue w_n.

    Returns:
    float: The eigenvalue λ_n.
    """
    return (2 * eta * variance) / (eta**2 * wn**2 + 1)

def eigenfunction(eta, wn, x, L):
    """
    Calculate the eigenfunction f_n(x).

    Parameters:
    eta (float): The parameter eta.
    wn (float): The eigenvalue w_n.
    x (float or array-like): The x values at which to evaluate the eigenfunction.
    L (float): The length parameter L.

    Returns:
    float or array-like: The value of the eigenfunction f_n(x).
    """
    factor = 1 / np.sqrt((eta**2 * wn**2 + 1) * L / 2 + eta)
    return factor *(eta * wn * np.cos(wn * x) + np.sin(wn * x))

# calculate the eigenvalue & eigenfunction
lambda_n = []
fn_values = np.zeros((L, nkl))
for i in range(nkl):
    wn = roots[i]
    eigen_value = eigenvalue(eta, variance, wn)
    lambda_n.append(eigen_value)
    eigen_vector = eigenfunction(eta, wn, x_values, L)
    fn_values[:,i] = eigen_vector

np.save(save_path + 'eigen_values', lambda_n)
np.save(save_path + 'eigen_vectors', fn_values)