"""
I want to try to replicate the results found in equation 17 of arXiv: 1902.06720

NOTE: This code will run significantly slower if using a CPU. It is recommended to use a GPU.
If no GPU is available, in the hyperparameters section it is recommended to reduce the number 
of repeat_exp and the maximum width of hidden layer.

**Usage:**

```bash 
python nconvergence_NTK.py 

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
from utils.helper_parse_ct_data import parse_ct_data, parse_ct_data_scaled

######################### Hyperparameters #########################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_identity_activation = False
Cw = 2
Cb = 0

num_data_points = 2
list_width_hidden_layer = [64, 128, 256, 512, 1024, 2048, 4096]
num_layers = 2

scale_learning_rate_tensor = False

num_networks_ensemble = 50  # 50 will take about an hour

num_train_steps = 1000 
epsilon = 1e-5
delay_train_steps = 20
###################################################################

def main():
    script_dir = os.path.dirname(__file__)  
    path_to_ct_data = 'utils/ct_data.npz'
    file_path = os.path.join(script_dir, path_to_ct_data) 

    # Obtain data that is L2 normalized
    X_train, X_val, X_test, y_train, y_val, y_test = parse_ct_data_scaled(file_path)
    num_inputs = X_train.shape[1]

    #  Scale learning rate tensor appropriately
    lambda_w_inputs = 1 / num_inputs if scale_learning_rate_tensor else 1
    lambda_b = 1

    # Convert to Torch tensors
    X_train = torch.from_numpy(X_train).float().to(device)
    X_val = torch.from_numpy(X_val).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    y_train = torch.from_numpy(y_train).float().to(device)
    y_val = torch.from_numpy(y_val).float().to(device)
    y_test = torch.from_numpy(y_test).float().to(device)

    # Slightly reduce X_train size
    X_train = X_train[:1000]
    y_train = y_train[:1000]

    X_reduced = X_train[:num_data_points]

    # Let's now track evolution of NTK. We wil track the evolution of the selected element
    time_evol_NTK = np.zeros((len(list_width_hidden_layer), num_networks_ensemble, num_train_steps // delay_train_steps))

    # We don't calculate H after every training step, but after every delay_train_steps training steps
    list_train_steps = np.arange(1, num_train_steps // delay_train_steps + 1) * delay_train_steps

    # We have to check that our learning rate is also lower than the critical learning rate. 
    lowest_critical_learning_rate = float('inf')

    for i,width_hidden_layer in enumerate(list_width_hidden_layer):
        lambda_w_hidden_layer = 1 / width_hidden_layer if scale_learning_rate_tensor else 1
        print(f"\nStarting experiments for width_hidden_layer = {width_hidden_layer} ...")

        for m in tqdm(range(num_networks_ensemble)):
            model = initiate_model(num_inputs, width_hidden_layer, num_layers, Cw, use_identity_activation, device)
            
            lambda_w, num_params = get_lambda_matrix_diagonal(lambda_w_inputs, lambda_w_hidden_layer, lambda_b, num_inputs, width_hidden_layer, num_layers)
            
            # We now train model
            H0 = obtain_H(model, X_reduced, lambda_w, num_data_points, num_params)
            
            critical_learning_rate = get_critical_learning_rate(H0)
            lowest_critical_learning_rate = min(lowest_critical_learning_rate, critical_learning_rate)
            
            for j in range(num_train_steps):
                train_model_one_step(model, X_train, y_train, epsilon, lambda_w_inputs, lambda_w_hidden_layer, lambda_b)
                    
                if j % delay_train_steps == 0:
                    H = obtain_H(model, X_reduced, lambda_w, num_data_points, num_params)
                    time_evol_NTK[i, m, j//delay_train_steps] = frobenius_norm(H - H0)
                    
                    critical_learning_rate = get_critical_learning_rate(H)
                    lowest_critical_learning_rate = min(lowest_critical_learning_rate, critical_learning_rate)
                
    print(f"\nLowest critical learning rate is {lowest_critical_learning_rate:0.5f}")
    if lowest_critical_learning_rate < epsilon:
        raise ValueError("Learning rate is too high! Theory requires it to be lower than the critical learning rate.")
    else:
        print("Learning rate is lower than the critical learning rate as requested by the theory. Computational results will hold.")
                
    # We take supremum and average over repeat_exp
    sup_time_evol_NTK = np.max(time_evol_NTK, axis=-1)                                           # (len(list_width_hidden_layer), num_networks_ensemble)
    average_sup_time_evol_NTK = np.mean(sup_time_evol_NTK, axis=-1)                              # (len(list_width_hidden_layer),)
    ste_sup_time_evol_NTK = np.std(sup_time_evol_NTK, axis=-1) / np.sqrt(num_networks_ensemble)  # (len(list_width_hidden_layer),)

    # Define power law
    def linear_fit(x, a, b):
        return a * x + b

    # Calculate weights for fitting in log space
    weights = 1.0 / (average_sup_time_evol_NTK*np.log(2)) * ste_sup_time_evol_NTK

    # Perform the curve fit with weights
    popt, pcov = curve_fit(linear_fit, np.log2(list_width_hidden_layer), np.log2(average_sup_time_evol_NTK), sigma=weights, absolute_sigma=True)

    std_slope = np.sqrt(pcov[0,0])
    print(f"\nOn the log scale, the slope is {popt[0]:0.5f} with a std of {std_slope:0.5f}")

    # Generate x values for plotting the fit
    x_fit = np.log2(np.logspace(np.min(np.log2(list_width_hidden_layer)), np.max(np.log2(list_width_hidden_layer)), 100, base=2))
    y_fit = linear_fit(x_fit, *popt)

    # Define function for log2 scaling the axes
    def set_log2_scale(ax):
        ax.set_xscale('log', base=2)
        ax.set_yscale('log', base=2)

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot the error bars
    ax.errorbar(list_width_hidden_layer, average_sup_time_evol_NTK, yerr=ste_sup_time_evol_NTK, fmt='.')
    ax.plot(2**x_fit, 2**y_fit, label=f'Linear fit α {popt[0]:.3f}n', linestyle='--', color='r')
    set_log2_scale(ax)

    plt.legend()
    plt.xlabel("Width of Hidden Layer")
    plt.ylabel(r"Supremum $||H_t - H_0||_F$")
    plt.title("Neural Tangent Kernel Evolution in Training")

    fig_path = "Data/evolution_NTK_normalized_x.png"

    # Create the directory if it doesn't exist
    os.makedirs(os.path.dirname(fig_path), exist_ok=True) 
    plt.savefig(fig_path, dpi=300, bbox_inches='tight') 

    print("\nFigure was saved to the Data folder.\n")
    plt.show()

if __name__ == "__main__":
    main()