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
num_inputs = 100
width_hidden_layer = 100
num_networks = int(2E4)
num_layers = 3
###################################################################

def main():
    x = torch.from_numpy(np.random.rand(num_inputs)).type(torch.FloatTensor).to(device)
    array_phi = np.zeros((num_networks*width_hidden_layer, num_layers))

    print("\nObtaining computational results...\n")
    for n in tqdm(range(num_networks)): 
        array_phi[n*width_hidden_layer:(n+1)*width_hidden_layer] = obtain_neuron_activation_all_layers_one_input(num_inputs, width_hidden_layer, num_layers, x) 
        
    array_phi_pow2 = array_phi**2
    array_phi_pow4 = array_phi**4

    k_l_numerical = np.mean(array_phi_pow2, axis=0)
    std_k_l_numerical = np.sqrt(np.var(array_phi_pow2, axis=0)) / np.sqrt(num_networks*width_hidden_layer)

    V_l_numerical = (np.mean(array_phi_pow4, axis=0) - 3*np.mean(array_phi_pow2, axis=0)**2)/3
    std_phi_pw2 = np.sqrt(np.var(array_phi_pow2, axis=0)) / np.sqrt(num_networks*width_hidden_layer)
    std_phi_pw4 = np.sqrt(np.var(array_phi_pow4, axis=0)) / np.sqrt(num_networks*width_hidden_layer)
    std_V_l_numerical = np.sqrt((std_phi_pw4/3)**2 + (std_phi_pw2)**2)
    
    print("\n####################### Computational Results #######################", "\n")
    for l in range(num_layers):
        print(f"K_{l+1} = {k_l_numerical[l]:.4f} +- {std_k_l_numerical[l]:.4f}")
    
    print("\n")
    for l in range(num_layers):
        print(f"V_{l+1} = {V_l_numerical[l]*width_hidden_layer:.4f} +- {std_V_l_numerical[l]*width_hidden_layer:.4f}")
    
    print("\n####################### Theoretical Results #######################", "\n")
    
    x = x.cpu().numpy()

    k0_1 = Cb + Cw * np.mean(np.square(x))
    k1_1 = 0
    V_1 = 0
    
    list_k = np.zeros((num_layers))
    list_V = np.zeros((num_layers))
    list_k[0] = k0_1
    list_V[0] = V_1

    for lp1 in range(2,num_layers+1):
        k_l = list_k[lp1-2]
        V_l = list_V[lp1-2]
        
        k_lp1 = Cb + Cw*expectation_gaussian_measure(lambda y: f(y)**2, 0, np.sqrt(k_l)) + Cw*V_l/8*expectation_gaussian_measure(fsq_d4, 0, np.sqrt(k_l)) * k_l**4
        list_k[lp1-1] = k_lp1
        
        # Let's now calculate V_lplus1
        chi_parallel_k_l = Cw / 2 * expectation_gaussian_measure(fsq_d2, 0, np.sqrt(k_l))
        diff_expectation_gaussian_measure = expectation_gaussian_measure(lambda y: f(y)**4, 0, np.sqrt(k_l)) - expectation_gaussian_measure(lambda y: f(y)**2, 0, np.sqrt(k_l))**2
        V_lp1 = chi_parallel_k_l**2 * V_l * k_l**4 + Cw**2 * diff_expectation_gaussian_measure
        list_V[lp1-1] = V_lp1 
    
    for l in range(num_layers):
        print(f"K_{l+1} = {list_k[l]:.4f}")
    
    print("\n")
    for l in range(num_layers):
        print(f"V_{l+1} = {list_V[l]:.4f}")
    print("\n")

if __name__ == "__main__":
    main()