import torch
import torch.nn as nn

# KLE function
def eigenvalue(eta, variance, wn):
    return (2 * eta * variance) / (eta**2 * wn**2 + 1)

def eigenfunction(eta, wn, x, L):
    factor = 1 / torch.sqrt((eta**2 * wn**2 + 1) * L / 2 + eta)
    return factor * (eta * wn * torch.cos(wn * x) + torch.sin(wn * x))

def compute_Ks(z, xi, eigenvalue, mu, eta, wn, L):
    eigen_function = eigenfunction(eta, wn, z, L)
    mu_lnKs = torch.full_like(z, fill_value=mu)
    sqrt_eigenvalue = torch.sqrt(eigenvalue).view(1, -1)
    KLE_result = torch.sum(eigen_function * sqrt_eigenvalue * xi, 
                           dim=1, keepdim=True)
    lnKs = mu_lnKs + KLE_result
    Ks = torch.exp(lnKs)
    return Ks
    
# PINN model
def initialize_weights(model):
    for layer in model.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)

class encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.Tanh()
        initialize_weights(self)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        return x

class modified_mlp(nn.Module):
    def __init__(self, input_dim, hidden_dim=81):
        super(modified_mlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()
        initialize_weights(self)

    def forward(self, x, enc1_out, enc2_out):
        h = self.activation(self.fc1(x))
        h = h * enc1_out + (1.0 - h) * enc2_out
        h = self.activation(self.fc2(h))
        h = h * enc1_out + (1.0 - h) * enc2_out
        h = self.activation(self.fc3(h))
        h = h * enc1_out + (1.0 - h) * enc2_out
        h = self.activation(self.fc4(h))
        h = h * enc1_out + (1.0 - h) * enc2_out
        h = self.fc5(h)
        return h

class random_coefficients(nn.Module):
    def __init__(self, nkl, device):
        super(random_coefficients, self).__init__()
        self.xi = nn.Parameter(torch.zeros((1, nkl), device=device))

    def forward(self, input_size):
        return self.xi.repeat(input_size, 1)

def normalize_tz(t, z):
    t_norm = 2*(t / 10) - 1
    z_norm = 2*((z + 99) / 99) - 1
    return t_norm, z_norm

class pinn(nn.Module):
    def __init__(self, nkl, device):
        super(pinn, self).__init__()
        self.device = device
        self.encoder_1 = encoder(2, 81)
        self.encoder_2 = encoder(2, 81)
        self.modified_mlp = modified_mlp(2)
        self.random_coefficients = random_coefficients(nkl, device)

    def forward(self, t, z):
        # normalize the input
        t_norm, z_norm = normalize_tz(t, z)
        # input & output
        nn_input = torch.cat([t_norm, z_norm], dim=-1)
        enc1_out = self.encoder_1(nn_input)
        enc2_out = self.encoder_2(nn_input)
        out_mlp = self.modified_mlp(nn_input, enc1_out, enc2_out)

        return -torch.exp(out_mlp)