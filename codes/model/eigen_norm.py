# Libraries
import numpy as np

# min-max norm
eigen_rd_pth = '.../Geo-PINN/data/eigens/eta=50_var=3/' # replace with the correct path
eigen_function = np.load(eigen_rd_pth + 'eigen_vectors.npy')
eigen_values = np.load(eigen_rd_pth + 'eigen_values.npy').reshape(1,-1)

save_pth = eigen_rd_pth + 'normalize/'

eigen = eigen_function * np.sqrt(eigen_values)
eigen_mean = np.mean(eigen,axis=0)
eigen_min = np.min(eigen,axis=0)
eigen_max = np.max(eigen,axis=0)
eigen_range = eigen_max - eigen_min

np.save(save_pth + 'eigen_min', eigen_min)
np.save(save_pth + 'eigen_range', eigen_range)