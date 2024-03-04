"""
Computes and compares theoretical and computational values for the observable K 
in the case where we have 2 input vectors.

**Usage:**

```bash 
python compute_K_2inputs.py 

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
from utils.helper_parse_ct_data import parse_ct_data

######################### Hyperparameters #########################
device = 'cpu'
use_identity_activation = False
Cw = 2
Cb = 0

width_hidden_layer = 100
num_networks_ensemble = int(1e5)
###################################################################

def main(): 
    script_dir = os.path.dirname(__file__)  
    path_to_ct_data = 'utils/ct_data.npz'
    file_path = os.path.join(script_dir, path_to_ct_data) 

    # Obtain data
    X_train, X_val, X_test, y_train, y_val, y_test = parse_ct_data(file_path)

    # The code is intended for obtaining observable K for two input data points and for a network
    # with a single hidden layer. 
    num_data_points = 2
    num_layers = 2

    X_reduced = X_train[:num_data_points]
    num_inputs = X_train.shape[1]
    
    array_phi_x1 = np.zeros((num_networks_ensemble, num_layers))
    array_phi_x2 = np.zeros_like(array_phi_x1)
    
    x1 = torch.tensor(X_reduced[0], dtype=torch.float).to(device)
    x2 = torch.tensor(X_reduced[1], dtype=torch.float).to(device)
    
    print("\nObtaining results for K for a network with 1 hidden layer.")
    print("\nObtaining computational results...\n")
    
    for n in tqdm(range(num_networks_ensemble)):
        array_phi_x1[n], array_phi_x2[n] = obtain_neuron_activation_all_layers_two_inputs(num_inputs, width_hidden_layer, x1, x2, num_layers, Cw, use_identity_activation, device)
    
    # Now we calculate array_phi for the hidden layer
    array_phi_2d_hiddenlayer = np.zeros((num_networks_ensemble, num_data_points, num_data_points))
    array_phi_2d_hiddenlayer[:,0,0] = array_phi_x1[:,0] ** 2
    array_phi_2d_hiddenlayer[:,1,1] = array_phi_x2[:,0] ** 2
    array_phi_2d_hiddenlayer[:,0,1] = array_phi_x1[:,0] * array_phi_x2[:,0]
    array_phi_2d_hiddenlayer[:,1,0] = array_phi_x2[:,0] * array_phi_x1[:,0]
    
    # Now we do the same for the output layer
    array_phi_2d_output = np.zeros_like(array_phi_2d_hiddenlayer)
    array_phi_2d_output[:,0,0] = array_phi_x1[:,1] ** 2
    array_phi_2d_output[:,1,1] = array_phi_x2[:,1] ** 2
    array_phi_2d_output[:,0,1] = array_phi_x1[:,1] * array_phi_x2[:,1]
    array_phi_2d_output[:,1,0] = array_phi_x2[:,1] * array_phi_x1[:,1]
    
    k_l_numerical_hiddenlayer = np.mean(array_phi_2d_hiddenlayer, axis=0)
    std_k_l_numerical_hiddenlayer = np.sqrt(np.var(array_phi_2d_hiddenlayer, axis=0)) / np.sqrt(num_networks_ensemble)
    
    k_l_numerical_output = np.mean(array_phi_2d_output, axis=0)
    std_k_l_numerical_output = np.sqrt(np.var(array_phi_2d_output, axis=0)) / np.sqrt(num_networks_ensemble)
    
    k_l_numerical = np.zeros((num_layers, num_data_points, num_data_points))
    k_l_numerical[0] = k_l_numerical_hiddenlayer
    k_l_numerical[1] = k_l_numerical_output
    
    std_k_l_numerical = np.zeros_like(k_l_numerical)
    std_k_l_numerical[0] = std_k_l_numerical_hiddenlayer
    std_k_l_numerical[1] = std_k_l_numerical_output
    
    
    print("\n####################### Computational Results #######################")
    print("\n K at Hidden Layer \n", k_l_numerical[0], "\n\nStandard Error\n", std_k_l_numerical[0], "\n\n")
    print("\n K at Output Layer \n", k_l_numerical[1], "\n\nStandard Error\n", std_k_l_numerical[1], "\n\n")
    
    
    # ############################ Theoretical ############################
    
    # In the 1D case, we calculate same thing but taking average
    k0_1 = Cb + Cw/num_inputs * np.einsum('im,km -> ik', X_reduced, X_reduced)
    deriv_handler = DerivativesActivationFunction(use_identity_activation)
    
    array_k0 = np.zeros((num_layers, num_data_points, num_data_points))
    array_k0[0] = k0_1

    for lp1 in range(2,num_layers+1):
        k0_l = array_k0[lp1-2]  # (2,2)
        
        for i in range(num_data_points):
            for j in range(num_data_points):
                # Let's first calculate k0 for the lth layer, given the l-1 layer values
                array_k0[lp1-1,i,j] = Cb + Cw * expectation_gaussian_measure_ndim(deriv_handler.f, deriv_handler.f, i, j, k0_l)

    print("####################### Theoretical Results #######################", "\n")
    print("\n K at Hidden Layer \n", array_k0[0])
    print("\n K at Output Layer \n", array_k0[1], "\n")
    
if __name__ == "__main__":
    main()