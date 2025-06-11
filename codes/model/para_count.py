import os
os.chdir('.../KLE-PINN-EC/codes') # replace with the correct path
import torch
import torch.nn as nn
import numpy as np
# models that need counting
from model import kle_pinn_ec

DEVICE = 'cpu'

mu_lnKs = 3.0
eta = 50
L = 100
variance = 3
nkl = 9

# initialize the model
nkl = 9
eigen_info = 'eta=' + str(eta) + '_var=' + str(variance) + '/'
eigen_path = '.../KLE-PINN-EC/data/eigens/' + eigen_info # replace with the correct path
wn_np = np.load(eigen_path + 'selected_wn.npy') # selected positive roots
wn = torch.tensor(wn_np, dtype=torch.float32).to(DEVICE)

model = kle_pinn_ec.pinn(nkl, eta, variance, wn, L, DEVICE).to(DEVICE)
print(model)

# Calculate the number of trainable parameters of the model
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Total trainable parameters: {count_trainable_parameters(model)}")