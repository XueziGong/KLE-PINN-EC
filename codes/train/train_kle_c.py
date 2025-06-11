# Libraries
import torch
from train import kle_c_train

# PyTorch version and GPU configuration
print("PyTorch version:", torch.__version__)
# check whether GPU is available
if torch.cuda.is_available():
    print("CUDA is available. GPU support is enabled.")
    print("Number of GPUs available:", torch.cuda.device_count())
    print("GPU Name:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available. GPU support is disabled.")
# device configuration
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Device: ', DEVICE)

# training loop
' Experimental setup '
mu_lnKs = 3.0
eta = 50
L = 100
variance = 3
nkl = 9

kle_c_train.loop(DEVICE, mu_lnKs, eta, L, variance, nkl, exp_design='benchmark', random_seed=1)