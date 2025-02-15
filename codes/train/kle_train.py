# Libraries
import numpy as np
import torch
import torch.nn as nn
from model import kle_pinn
from model import MVG_model
from tqdm import tqdm

# training loop function
def loop(DEVICE, mu_lnKs, eta, L, variance, nkl, exp_design, random_seed):

    model_info = 'kle_pinn/'
    
    ' Random seed '
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    
    print('Current exp design:', exp_design)
    print('Current random seed:', random_seed)
    
    ' File path '
    eigen_info = 'eta=' + str(eta) + '_var=' + str(variance) + '/'
    main_path = '.../Geo-PINN/results/inverse/' + exp_design + '/' # replace with the correct path
    eigen_path = '.../Geo-PINN/data/eigens/' + eigen_info # replace with the correct path
    training_data_path = '.../Geo-PINN/data/' + exp_design + '/' # replace with the correct path
    group_path = str(random_seed) + '/'
    save_path = main_path + model_info + group_path

    ' Training dataset '
    # KLE
    wn_np = np.load(eigen_path + 'selected_wn.npy') # selected positive roots
    wn = torch.tensor(wn_np, dtype=torch.float32).to(DEVICE)
    eigen_values = kle_pinn.eigenvalue(eta, variance, wn).reshape(-1,1)
    
    # collocation points
    n_samples = 4000
    time_range = [0, 10]
    space_range = [-99, 0]
    
    # observation points
    t_list = []
    for i in range(101):
        for j in range(6):
            t_list.append(i/10)
    t_obs_np = np.array(t_list).reshape(-1, 1)
    t_obs = torch.tensor(t_obs_np, dtype=torch.float32, requires_grad=True).to(DEVICE)
    
    z_list = []
    for i in range(101):
        for j in range(6):
            z_list.append(-5-18*j)
    z_obs_np = np.array(z_list).reshape(-1, 1)
    z_obs = torch.tensor(z_obs_np, dtype=torch.float32, requires_grad=True).to(DEVICE)
    
    # observation data
    obs_theta_np = np.load(training_data_path + 'obs_theta.npy')
    obs_theta = torch.tensor(obs_theta_np, dtype=torch.float32).to(DEVICE)
    obs_theta = obs_theta.T.flatten().reshape(-1,1)

    # build the model
    pinn = kle_pinn.pinn(nkl, DEVICE).to(DEVICE)

    num_epochs = 100000 # total epochs
    mse_loss = nn.MSELoss()

    # optimizer info
    lr = 1e-3
    optimizer_adam = torch.optim.Adam(pinn.parameters(), lr=lr)
    loss_history = []
    loss_physics_history = []
    loss_data_history = []
    
    # lambda of the loss term
    lambda_physics = 1.0
    lambda_theta = 0.1
    
    def grad(outputs, inputs):
        return torch.autograd.grad(outputs, inputs, 
                                   grad_outputs=torch.ones_like(outputs), 
                                   create_graph=True)[0]
    
    def compute_loss():
        # forward Propagation
        xi_pred_col = pinn.random_coefficients.xi.repeat(z_col.size(0), 1)    
        Ks_col = kle_pinn.compute_Ks(z_col, xi_pred_col, eigen_values,
                                     mu_lnKs, eta, wn, L)
        psi_col = pinn(t_col, z_col)
        theta_col = MVG_model.theta(psi_col)
        Kr_col = MVG_model.Kr(theta_col)
        K_col = MVG_model.K(Ks_col, Kr_col)
        psi_pred = pinn(t_obs, z_obs)
        theta_pred = MVG_model.theta(psi_pred)
    
        # compute the loss
        theta_t = grad(theta_col, t_col)
        K_z = grad(K_col, z_col)
        psi_z = grad(psi_col, z_col)
        psi_zz = grad(psi_z, z_col)
        residual_physics = theta_t - K_z * psi_z - K_col * psi_zz - K_z
        loss_physics = torch.mean(torch.square(residual_physics))
        loss_theta = mse_loss(theta_pred, obs_theta)
        
        return loss_theta, loss_physics

    for epoch in tqdm(range(num_epochs)):
        pinn.train()
        # collocation points
        t_col = (torch.rand(n_samples, device=DEVICE) * 
                 (time_range[1] - time_range[0]) + time_range[0]).reshape(-1,1).requires_grad_()
        z_col = (torch.rand(n_samples, device=DEVICE) * 
                 (space_range[1] - space_range[0]) + space_range[0]).reshape(-1,1).requires_grad_()
        
        loss_theta, loss_physics = compute_loss()
        loss_total = lambda_theta * loss_theta + lambda_physics*loss_physics
        
        # optimize
        optimizer_adam.zero_grad()
        loss_total.backward()
        optimizer_adam.step()
        loss_history.append(loss_total.item())
        loss_physics_history.append(loss_physics.item())
        loss_data_history.append(loss_theta.item())

    # save
    np.save(save_path + 'loss_history', loss_history)
    np.save(save_path + 'loss_physics_history', loss_physics_history)
    np.save(save_path + 'loss_data_history', loss_data_history)
    torch.save(pinn.state_dict(), save_path + 'model_params.pth')