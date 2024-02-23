"""
Computes and compares experimental and theoretical values for the Neural Tangent Kernel (NTK).

**Usage:**

```bash 
python compute_NTK.py 

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

script_dir = os.path.dirname(__file__)  
path_to_ct_data = 'utils/ct_data.npz'
file_path = os.path.join(script_dir, path_to_ct_data) 

# Obtain data
X_train, X_val, X_test, y_train, y_val, y_test = parse_ct_data(file_path)

# Convert to Torch tensors
X_train = torch.from_numpy(X_train).float().to(device)
X_val = torch.from_numpy(X_val).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_val = torch.from_numpy(y_val).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)


######################### Hyperparameters #########################
num_data_points = 2
num_inputs = X_train.shape[1]
width_hidden_layer = 100

lambda_w_inputs = 1 / num_inputs
lambda_w_hidden_layer = 1 / width_hidden_layer
lambda_b = 1

repeat_exp = 1000
###################################################################


def main():
    X_reduced = X_train[:num_data_points]
    num_inputs = X_reduced.shape[1]

    # Let's obtain H0_calc with error bars
    array_H0_calc = torch.zeros((repeat_exp, num_data_points, num_data_points)).to(device)
    for m in tqdm(range(repeat_exp)):
        model = initiate_model(num_inputs, width_hidden_layer)
        if m==0:
            print(model)
            print("\nObtaining computational results for the Neural Tangent Kernel...\n")
        
        lambda_matrix_diagonal, num_params = get_lambda_matrix_diagonal(lambda_w_inputs, lambda_w_hidden_layer, lambda_b, num_inputs, width_hidden_layer)

        array_H0_calc[m] = obtain_H(model, X_reduced, lambda_matrix_diagonal, num_data_points, num_params)

    mean_H0_calc = torch.mean(array_H0_calc, axis=0)
    std_H0_calc = torch.sqrt(torch.var(array_H0_calc, axis=0)/repeat_exp)

    print("\n####################### Computational Results #######################")
    print("\n Neural Tangent Kernel at Last Layer \n", mean_H0_calc.cpu().numpy(), "\n\nStandard Error\n", std_H0_calc.cpu().numpy(), "\n\n")
    
    print("####################### Theoretical Results #######################", "\n")

    X_reduced = X_reduced.cpu().numpy()
    dotted_inputs = np.einsum('im,km -> ik', X_reduced, X_reduced)

    # X is of size (num_data_points, 384)
    K_1 = Cb + Cw/num_inputs * dotted_inputs
    theta_1 = lambda_b + lambda_w_inputs * dotted_inputs  #lambda_b + lambda_w/num_inputs * dotted_inputs 

    expectation_func = np.zeros_like(theta_1)
    expectation_deriv = np.zeros_like(theta_1)

    for i in range(num_data_points):
        for j in range(num_data_points):
            expectation_func[i,j]  = expectation_gaussian_measure_ndim(f,    f,    i, j, K_1)
            expectation_deriv[i,j] = expectation_gaussian_measure_ndim(f_d1, f_d1, i, j, K_1)
    
    H_2 = lambda_b + lambda_w_hidden_layer * expectation_func * width_hidden_layer + Cw * expectation_deriv * theta_1  # lambda_b + lambda_w * expectation_func + Cw * expectation_deriv * theta_1
    print("Neural Tangent Kernel at Last Layer:\n", H_2, "\n")

    H_identity = lambda_b + lambda_w_hidden_layer * K_1 * width_hidden_layer + Cw * theta_1  # lambda_b + lambda_w * K_1 + Cw * theta_1 
    #print("\Neural Tangent Kernel at Last Layer for the Identity Activation:\n", H_identity, "\n")

if __name__ == "__main__":
    main()