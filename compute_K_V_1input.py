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

import os
import sys
from utils.helper_computation import *
from utils.helper_theory import *

######################### Hyperparameters #########################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_identity_activation = False
Cw = 2
Cb = 0

num_inputs = 100
width_hidden_layer = 100
num_networks_ensemble = int(1E3)
num_layers = 4
###################################################################

def main():
    x = torch.from_numpy(np.random.rand(num_inputs)).type(torch.FloatTensor).to(device)
    array_phi = np.zeros((num_networks_ensemble, num_layers))

    print("\nObtaining computational results...\n")
    for n in tqdm(range(num_networks_ensemble)): 
        array_phi[n] = obtain_neuron_activation_all_layers_one_input(num_inputs, width_hidden_layer, num_layers, x, Cw, use_identity_activation, device)
        
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
    list_k0, list_k1, list_V = compute_KV_1input(x, num_inputs, width_hidden_layer, num_layers, Cb, Cw)
    
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

if __name__ == "__main__":
    main()