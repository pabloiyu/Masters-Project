"""
Computes and compares experimental and theoretical values for K in the case where 
we have one input vector.

**Usage:**

```bash 
python compute_K_V_1input.py 

"""
import numpy as np 
import torch
import torch as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.stats import norm, entropy
from scipy.integrate import simpson

import os
import sys
from utils.helper_computation import *
from utils.helper_theory import *

######################### Hyperparameters #########################
device = 'cpu'
use_identity_activation = False
Cw = 2
Cb = 0

num_inputs = 100
width_hidden_layer = 100
num_networks_ensemble = int(4e4)
num_layers = 4

use_numpy_implementation = True
###################################################################

def main():
    x = torch.from_numpy(np.random.rand(num_inputs)).type(torch.FloatTensor).to(device)
    # From each network in the ensemble, and at each layer, we store the activation of a single neuron
    array_phi = np.zeros((num_networks_ensemble, num_layers))
    
    if use_numpy_implementation:
        print("\nUsing Numpy Implementation...\n")
        array_phi_all_activations_hiddenlayers, array_phi_all_activations_output = obtain_neuron_activation_all_layers_one_input_numpy(num_inputs, width_hidden_layer, num_layers, num_networks_ensemble, x.numpy(), Cw, Cb, use_identity_activation=False, device='cpu')
        # array_phi_all_activations_hiddenlayers.size = (num_networks_ensemble, width_hidden_layer, num_layers-1)
        # array_phi_all_activations_output.size       = (num_networks_ensemble, 1)
        
        # We now pick only the first neuron in each hidden layer
        array_phi[:, :-1] = array_phi_all_activations_hiddenlayers[:,0,:]
        array_phi[:, -1] = array_phi_all_activations_output[:, 0]
        
        # We now reshape array_phi_all_activations_hiddenlayers to have the shape (num_networks_ensemble*width_hidden_layer, num_layers-1)
        array_phi_all_activations_hiddenlayers = array_phi_all_activations_hiddenlayers.reshape((num_networks_ensemble*width_hidden_layer, num_layers-1))
        
    else:
        # We save the activation of all neurons in the hidden layers and the output layer 
        array_phi_all_activations_hiddenlayers = np.zeros((num_networks_ensemble*width_hidden_layer, num_layers-1))
        array_phi_all_activations_output = np.zeros((num_networks_ensemble, 1))

        print("\nObtaining computational results...\n")
        for n in tqdm(range(num_networks_ensemble)): 
            array_phi[n], all_activations_different_layers = obtain_neuron_activation_all_layers_one_input(num_inputs, width_hidden_layer, num_layers, x, Cw, use_identity_activation, device)
            
            array_phi_all_activations_hiddenlayers[n*width_hidden_layer:(n+1)*width_hidden_layer] = np.array(all_activations_different_layers[:-1]).T
            array_phi_all_activations_output[n] = all_activations_different_layers[-1]
        
    # As there is only one output neuron, we can remove the last dimension of the output array
    array_phi_all_activations_output.squeeze()
        
    array_phi_pow2 = array_phi**2
    array_phi_pow4 = array_phi**4

    k_l_numerical = np.mean(array_phi_pow2, axis=0)
    std_k_l_numerical = np.sqrt(np.var(array_phi_pow2, axis=0)) / np.sqrt(num_networks_ensemble)

    V_l_numerical = (np.mean(array_phi_pow4, axis=0) - 3*np.mean(array_phi_pow2, axis=0)**2)/3
    std_phi_pw2 = np.sqrt(np.var(array_phi_pow2, axis=0)) / np.sqrt(num_networks_ensemble)
    std_phi_pw4 = np.sqrt(np.var(array_phi_pow4, axis=0)) / np.sqrt(num_networks_ensemble)
    std_V_l_numerical = np.sqrt((std_phi_pw4/3)**2 + (std_phi_pw2)**2)
    
    print("\n####################### Computational Results #######################", "\n")
    for l in range(num_layers):
        print(f"K_{l+1} = {k_l_numerical[l]:.6f} +- {std_k_l_numerical[l]:.6f}")
    
    print("\n")
    for l in range(num_layers):
        print(f"V_{l+1} = {V_l_numerical[l]:.6f} +- {std_V_l_numerical[l]:.6f}")
    
    print("\n####################### Theoretical Results #######################", "\n")
    
    x = x.cpu().numpy()
    list_k0, list_k1, list_V = compute_KV_1input(x, num_inputs, width_hidden_layer, num_layers, Cb, Cw, use_identity_activation)
    
    # We have to divide K_1 and V by the width of the hidden layer as these theoretical equations
    # describe their vl
    for l in range(num_layers):
        print(f"K_LO_{l+1} = {list_k0[l]:.6f}")
        
    print("\n")
    for l in range(num_layers):
        print(f"K_{l+1} = {list_k0[l] + list_k1[l]/width_hidden_layer:.6f}")
    
    print("\n")
    for l in range(num_layers):
        print(f"V_{l+1} = {list_V[l]/width_hidden_layer:.6f}")
    print("\n")
    
    # We compute the KL divergence between the empirical PDF of the activation of the neurons
    # in the hidden layers
    print("\n####################### HISTOGRAMS #######################", "\n")
    
    k_l_numerical = np.mean(array_phi_all_activations_hiddenlayers**2, axis=0)
    for l in range(num_layers-1):
        x_grid = np.linspace(-4, 4, 5000) 
        
        # Theoretical Gaussian
        theoretical_var = list_k0[l] + list_k1[l]/width_hidden_layer
        theoretical_V = list_V[l]/width_hidden_layer

        theoretical_pdf_unnormalised = probability_distribution_phi(x_grid, theoretical_var, theoretical_V)
        Z = simpson(theoretical_pdf_unnormalised, x=x_grid)
        theoretical_pdf_vals = theoretical_pdf_unnormalised / Z

        # Estimate empirical PDF (KDE)
        empirical_var = k_l_numerical[l]
        empirical_V = V_l_numerical[l]
        
        empirical_pdf_unnormalised = probability_distribution_phi(x_grid, empirical_var, empirical_V)
        Z = simpson(empirical_pdf_unnormalised, x=x_grid)
        empirical_pdf_vals = empirical_pdf_unnormalised / Z

        kl_div = entropy(theoretical_pdf_vals, empirical_pdf_vals)
        print(f"KL Divergence at Hidden Layer {l+1}: {kl_div:.10f}") 
    
    # We plot the histograms of the activation of the neurons in the hidden layers and the output layer
    # Calculate the number of rows and columns needed
    num_plots = num_layers - 1
    num_rows = 1 if num_plots <= 2 else 2
    num_cols = num_plots if num_plots <= 2 else (num_plots + 1) // 2

    # Create the figure and axes
    if num_plots == 2:
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 7))
    else:
        fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 13))

    # Reshape axs to handle all cases
    axs = axs.ravel() if num_plots > 1 else np.array([axs])

    fig.suptitle('Pre-Activation Distributions Across Hidden Layers', fontsize=17)
    fig.subplots_adjust(hspace=.25, top=0.91)

    # Iterate over the layers and plot
    for l, ax in enumerate(axs):
        if l >= num_plots:
            ax.axis('off')
            continue

        x_grid = np.linspace(min(array_phi_all_activations_hiddenlayers[:,l]), max(array_phi_all_activations_hiddenlayers[:,l]), 1000) 

        # We now plot leading order theoretical distribution, which is a Gaussian.
        theoretical_var = list_k0[l]
        theoretical_pdf = norm(loc=0, scale=np.sqrt(theoretical_var))
        theoretical_pdf_vals = theoretical_pdf.pdf(x_grid)
        ax.plot(x_grid, theoretical_pdf_vals, linestyle='--', color='crimson', markersize=0, label='LO Distribution', linewidth=3.5)
        
        # The next to leading order distribution is no longer just a Gaussian as it has a exp(x^4) term
        theoretical_var = list_k0[l] + list_k1[l]/width_hidden_layer
        theoretical_V = list_V[l]/width_hidden_layer
        theoretical_pdf_unnormalised = probability_distribution_phi(x_grid, theoretical_var, theoretical_V)
        Z = simpson(theoretical_pdf_unnormalised, x=x_grid)
        theoretical_pdf_vals = theoretical_pdf_unnormalised / Z
        ax.plot(x_grid, theoretical_pdf_vals, linestyle='--', color='#EAAA00', markersize=0, label='LO + O(1/n) Distribution', linewidth=3.5)
        
        ax.hist(array_phi_all_activations_hiddenlayers[:,l], bins=300, density=True)
        ax.legend(fontsize=13)
        ax.set_xlabel('Pre-Activation Values', fontsize=14)
        ax.set_ylabel('Probability Density', fontsize=14)
        ax.set_title(f"Hidden Layer {l+1}", fontsize=16)
        
        ax.tick_params(axis='both', labelsize=12)
    
    fig_path = f"Data/histograms_neuron_activation_{width_hidden_layer}neurons_wide_{num_networks_ensemble}networks.png"  

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)  
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print("\nFigure was saved to the Data folder.\n")
    plt.show()
    

if __name__ == "__main__":
    main()