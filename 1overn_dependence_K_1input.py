"""
The computational value for K approximates both the leading order term and O(1/n^p) 
corrections. To differentiate these contributions, we calculate K for various values
of n and plot K against 1/n. Fitting a straight line to this plot yields the leading
order coefficient (K_0) as the intercept and the O(1/n) contribution (K_1) as the slope.

**Usage:**

```bash 
python 1overn_dependence_K_1input.py 

"""
import numpy as np 
import torch
import torch as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from scipy.optimize import curve_fit

import os
import sys
from utils.helper_computation import *
from utils.helper_theory import *

######################### Hyperparameters #########################
device = 'cpu'
use_identity_activation = False
Cw = 2
Cb = 0

num_inputs = 10
list_width_hidden_layer = [10, 20, 40, 60, 80, 100, 150, 200] 
num_networks_ensemble = int(2e4) 
num_layers = 3
###################################################################

def main():
    reciprocal_width = 1 / np.array(list_width_hidden_layer)
    
    x = torch.from_numpy(np.random.rand(num_inputs)).type(torch.FloatTensor).to(device)
    
    list_k_l_numerical = np.zeros((len(list_width_hidden_layer), num_layers))
    list_ste_k_l_numerical = np.zeros_like(list_k_l_numerical)
    
    for i,width_hidden_layer in enumerate(list_width_hidden_layer):
        array_phi = np.zeros((num_networks_ensemble, num_layers))

        print(f"\nStarting experiments for width_hidden_layer = {width_hidden_layer} ...")
        for n in tqdm(range(num_networks_ensemble)): 
            array_phi[n] = obtain_neuron_activation_all_layers_one_input(num_inputs, width_hidden_layer, num_layers, x, Cw, use_identity_activation, device)
        
        array_phi_pow2 = array_phi**2

        k_l_numerical = np.mean(array_phi_pow2, axis=0)
        std_k_l_numerical = np.sqrt(np.var(array_phi_pow2, axis=0)) / np.sqrt(num_networks_ensemble)
        
        list_k_l_numerical[i] = k_l_numerical
        list_ste_k_l_numerical[i] = std_k_l_numerical
    
    # Define power law
    def linear_fit(x, a, b):
        return a * x + b

    # Perform the curve fit with weights
    popt, pcov = curve_fit(linear_fit, reciprocal_width, list_k_l_numerical[:,-1], sigma=list_ste_k_l_numerical[:,-1], absolute_sigma=True)

    # Generate x values for plotting the fit
    x_fit = np.linspace(min(reciprocal_width), max(reciprocal_width), 1000)
    y_fit = linear_fit(x_fit, *popt)    
    
    x = x.cpu().numpy()
    list_k0, list_k1, list_V = compute_KV_1input(x, num_inputs, width_hidden_layer, num_layers, Cb, Cw, use_identity_activation)
    
    print("\n   k = k0 + k1/n \nThe intercept corresponds to k0 and the slope corresponds to k1.")
    
    print("\n####################### Computational Results #######################", "\n")
    
    print(f"Intercept: {popt[1]:.6f} +- {np.sqrt(pcov[1,1]):.6f}")
    print(f"Slope:     {popt[0]:.6f} +- {np.sqrt(pcov[0,0]):.6f}")
    
    print("\n####################### Theoretical Results #######################", "\n")
    
    print(f"Intercept: {list_k0[-1]:.6f}")
    print(f"Slope:     {list_k1[-1]:.6f}")
    print("\n")
    
    plt.errorbar(reciprocal_width, list_k_l_numerical[:,-1], yerr=list_ste_k_l_numerical[:,-1], fmt='.')
    plt.plot(x_fit, y_fit, linestyle='--', color='g', label="Computational Prediction")
    plt.plot(x_fit, linear_fit(x_fit, list_k1[-1], list_k0[-1]), linestyle='--', color='r', label="Theoretical Prediction")
    plt.legend()
    plt.xlabel(f'1/n')
    plt.ylabel(r'$K^{(L)}$')
    plt.title(r'Last-Layer Observable K as a Function of Inverse Network Size')
    
    fig_path = "Data/ndependence_K_1input.png"

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)  
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.show()
    

if __name__ == "__main__":
    main()