"""
Computes and compares experimental and theoretical values for the Neural Tangent Kernel (NTK) at
initialisation.

**Usage:**

```bash 
python compute_NTK_initialisation.py 

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

num_data_points = 2
width_hidden_layer = 100
num_layers = 5

scale_learning_rate_tensor = True

num_networks_ensemble = int(1e5)
###################################################################

def main():
    script_dir = os.path.dirname(__file__)  
    path_to_ct_data = 'utils/slice_localization_data.csv'
    file_path = os.path.join(script_dir, path_to_ct_data) 

    # Obtain data
    X_train, _, _, _, _, _ = parse_ct_data(file_path)
    num_inputs = X_train.shape[1]
    X_train = torch.from_numpy(X_train).float().to(device)

    #  Scale learning rate tensor appropriately
    lambda_w_inputs = 1 / num_inputs if scale_learning_rate_tensor else 1
    lambda_w_hidden_layer = 1 / width_hidden_layer if scale_learning_rate_tensor else 1
    lambda_b = 1
    
    # Define a reduced dataset
    X_reduced = X_train[:num_data_points]

    print("\nObtaining computational results for the Neural Tangent Kernel...\n")

    # Let's obtain H0_calc with error bars
    array_H0_calc = torch.zeros((num_networks_ensemble, num_data_points, num_data_points)).to(device)
    for m in tqdm(range(num_networks_ensemble)):
        model = initiate_model(num_inputs, width_hidden_layer, num_layers, Cw, use_identity_activation, device)            
        
        lambda_matrix_diagonal, num_params = get_lambda_matrix_diagonal(lambda_w_inputs, lambda_w_hidden_layer, lambda_b, num_inputs, width_hidden_layer, num_layers)

        array_H0_calc[m] = obtain_H(model, X_reduced, lambda_matrix_diagonal, num_data_points, num_params)

    mean_H0_calc = torch.mean(array_H0_calc, axis=0)
    std_H0_calc = torch.sqrt(torch.var(array_H0_calc, axis=0)/num_networks_ensemble)

    print("\n####################### Computational Results #######################")
    print("\n Neural Tangent Kernel at Last Layer \n", mean_H0_calc.cpu().numpy(), "\n\nStandard Error\n", std_H0_calc.cpu().numpy(), "\n\n")
    
    print("####################### Theoretical Results #######################", "\n")

    H_2 = compute_theoretical_prediction_NTK_initialisation(X_reduced.cpu().numpy(), num_inputs, width_hidden_layer, lambda_w_inputs, lambda_w_hidden_layer, lambda_b, Cb, Cw,  use_identity_activation)
    print("Neural Tangent Kernel at Last Layer:\n", H_2, "\n")

if __name__ == "__main__":
    main()