"""
Computes and compares experimental and theoretical values for the Neural Tangent Kernel (NTK)
at initialisation. Computes H for different n's and plots it against 1/n. The intercept will 
be the leading order contribution. The slope will be the O(1/n) correction.

**Usage:**

```bash 
python 1overn_dependence_NTK_initialisation.py 

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
from utils.helper_parse_ct_data import parse_ct_data

######################### Hyperparameters #########################
device = 'cpu'
use_identity_activation = False
Cw = 2
Cb = 0

scale_learning_rate_tensor = True

num_inputs = 10
num_data_points = 2

list_width_hidden_layer = [20, 40, 60, 80, 100, 150, 200] 
num_networks_ensemble = int(5e4) 
num_layers = 2
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
    lambda_b = 1

    reciprocal_width = 1 / np.array(list_width_hidden_layer)
    
    X = X_train[:num_data_points]
    
    # We only store the NTK at the singular output neuron. For each network in the ensemble, and for each width,
    # we will store the NTK at the singular output neuron.
    list_NTK_numerical = np.zeros((num_networks_ensemble, len(list_width_hidden_layer), num_data_points, num_data_points))

    for i,width_hidden_layer in enumerate(list_width_hidden_layer):
        # Knowing the width, scale the learning rate tensor appropriately for the hidden layer
        lambda_w_hidden_layer = 1 / width_hidden_layer if scale_learning_rate_tensor else 1

        print(f"\nStarting experiments for width_hidden_layer = {width_hidden_layer} ...")
        for n in tqdm(range(num_networks_ensemble)): 
            model = initiate_model(num_inputs, width_hidden_layer, num_layers, Cw, use_identity_activation, device)
        
            lambda_matrix_diagonal, num_params = get_lambda_matrix_diagonal(lambda_w_inputs, lambda_w_hidden_layer, lambda_b, num_inputs, width_hidden_layer, num_layers)

            list_NTK_numerical[n,i] = obtain_H(model, X, lambda_matrix_diagonal, num_data_points, num_params)
        
    # We now average over the ensemble of networks. We also compute the standard error of the mean.
    average_NTK_numerical = np.mean(list_NTK_numerical, axis=0)                              # [num_widths, num_data_points, num_data_points]
    ste_NTK_numerical = np.std(list_NTK_numerical, axis=0) / np.sqrt(num_networks_ensemble)  # [num_widths, num_data_points, num_data_points]
    
    print("\nComputing theoretical prediction for the NTK at initialisation ...")
    
    # We now compute the expected theoretical result for the NTK.
    H_2_theoretical = compute_theoretical_prediction_NTK_initialisation(X.cpu().numpy(), num_inputs, width_hidden_layer, lambda_w_inputs, lambda_w_hidden_layer, lambda_b, Cb, Cw,  use_identity_activation)
    
    # Define linear fit 
    def linear_fit(x, a, b):
        return a * x + b
    
    fig, axs = plt.subplots(num_data_points, num_data_points, figsize=(18, 13))
    fig.suptitle('Last-Layer Observable NTK as a Function of Inverse Network Size', fontsize=17)
    fig.subplots_adjust(hspace=.21*num_data_points/2, top=0.92)
    
    # We now perform the fit for every element of the NTK. We will plot and save a graph for each.
    for i in range(num_data_points):
        for j in range(num_data_points):
            popt, pcov = curve_fit(linear_fit, reciprocal_width, average_NTK_numerical[:,i,j], sigma=ste_NTK_numerical[:,i,j], absolute_sigma=True)
            
            # Generate x values for plotting the fit
            x_fit = np.linspace(min(reciprocal_width), max(reciprocal_width), 1000)
            y_fit = linear_fit(x_fit, *popt) 
    
            print("\n   k = k0 + k1/n \nThe intercept corresponds to k0 and the slope corresponds to k1.")
            
            print(f"\n####################### Computational Results | Element ({i},{j}) #######################", "\n")
            
            print(f"Intercept: {popt[1]:.6f} +- {np.sqrt(pcov[1,1]):.6f}")
            print(f"Slope:     {popt[0]:.6f} +- {np.sqrt(pcov[0,0]):.6f}")
            
            print(f"\n####################### Theoretical Results | Element ({i},{j}) #######################", "\n")
            
            print(f"Intercept: {H_2_theoretical[i,j]:.6f}")
            print(f"Slope:     0")
            print("\n")
            
            axs[i,j].errorbar(reciprocal_width, average_NTK_numerical[:,i,j], yerr=ste_NTK_numerical[:,i,j], fmt='.')
            axs[i,j].plot(x_fit, y_fit, linestyle='--', color='g', label="Computational Fit")
            axs[i,j].plot(x_fit, linear_fit(x_fit, 0, H_2_theoretical[i,j]), linestyle='--', color='r', label="Theoretical Fit")
            axs[i,j].legend(fontsize=13)
            axs[i,j].set_xlabel(f'1/n', fontsize=14)
            axs[i,j].set_ylabel(f'Neural Tangent Kernel', fontsize=14)
            axs[i,j].set_title(f'Element ({i},{j})', fontsize=16)
            
    fig_path = "Data/1overn_dependence_NTK_initialisation.png"  

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)  
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print("\nFigure was saved to the Data folder.\n")
    plt.show()
                

if __name__ == "__main__":
    main()