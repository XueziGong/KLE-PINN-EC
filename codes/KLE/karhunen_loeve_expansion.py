# Libraries
import numpy as np

# Covariance matrix
# Assuming parameters
n = 100  # number of nodes
mu = 3  # mean
eta = 50 # correlation length
variance = 3 # sigma^2

# Generating random numbers
random_seed = 20241
np.random.seed(random_seed)
xi = np.random.normal(0, 1, size=n)
nkl = 9
xi = xi[:nkl,]
eigen_path_1 = '.../Geo-PINN/data/eigens/' # replace with the correct path
eigen_path_2 = 'eta=' + str(eta) + '_var=' + str(variance) + '/'
eigen_path = eigen_path_1 + eigen_path_2
eigenvalues = np.load(eigen_path + 'eigen_values.npy')
eigenvectors = np.load(eigen_path + 'eigen_vectors.npy')

# Reduction of soil parameter fields using K-L expansion
lnKs = np.full(n, mu)  # initialised to mean
for i in range(nkl):
    lnKs = lnKs + np.sqrt(eigenvalues[i]) * eigenvectors[:, i] * xi[i]
print("Reduced soil parameter fields --- lnKs：", lnKs)

save_path_main = '.../Geo-PINN/reference_para/seed_' + str(random_seed) + '/' # replace with the correct path
save_path = save_path_main + eigen_path_2
np.save(save_path_main + 'xi', xi)
np.save(save_path + 'lnKs_reference', lnKs)

# calculation of beta
def cumulated_eigenvalues(eigenvalues,nkl):
    alpha = 0
    for i in range(nkl):
        alpha += eigenvalues[i]
    return alpha
beta = cumulated_eigenvalues(eigenvalues, nkl)/(variance*n)
print(beta)
