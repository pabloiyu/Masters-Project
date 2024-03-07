"""
Computes and compares experimental and theoretical values for the Neural Tangent Kernel (NTK).
Tracks their evolution over training steps.

I am trying to replicate Figure 10 of arXiv: 1909.11304 

IN PROGRESS

**Usage:**

```bash 
python evolution_NTK.py 

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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_identity_activation = False
Cw = 2

num_data_points = 2
width_hidden_layer = 128
num_layers = 2

scale_learning_rate_tensor = True

num_networks_ensemble = 10

num_train_steps = 1000
epsilon = 1e-1
delay_train_steps = 20
###################################################################

script_dir = os.path.dirname(__file__)  
path_to_ct_data = 'utils/slice_localization_data.csv'
file_path = os.path.join(script_dir, path_to_ct_data) 

# Obtain data
X_train, X_val, X_test, y_train, y_val, y_test = parse_ct_data(file_path)
num_inputs = X_train.shape[1]

#  Scale learning rate tensor appropriately
lambda_w_inputs = 1 / num_inputs if scale_learning_rate_tensor else 1
lambda_w_hidden_layer = 1 / width_hidden_layer if scale_learning_rate_tensor else 1
lambda_b = 1

# Convert to Torch tensors
X_train = torch.from_numpy(X_train).float().to(device)
X_val = torch.from_numpy(X_val).float().to(device)
X_test = torch.from_numpy(X_test).float().to(device)
y_train = torch.from_numpy(y_train).float().to(device)
y_val = torch.from_numpy(y_val).float().to(device)
y_test = torch.from_numpy(y_test).float().to(device)

# Slighlty reduce X_train size
X_train = X_train[:1000]
y_train = y_train[:1000]

X_reduced = X_train[:num_data_points]

selected_element_NTK = (1,0)

# Let's now track evolution of NTK. We wil track the evolution of the selected element
time_evol_NTK = np.zeros((num_networks_ensemble, num_train_steps // delay_train_steps))

# We don't calculate H after every training step, but after every delay_train_steps training steps
list_train_steps = np.arange(1, num_train_steps // delay_train_steps + 1) * delay_train_steps

print("Starting experiments...")
for m in tqdm(range(num_networks_ensemble)):
    model = initiate_model(num_inputs, width_hidden_layer, num_layers, Cw, use_identity_activation, device)
    
    lambda_w, num_params = get_lambda_matrix_diagonal(lambda_w_inputs, lambda_w_hidden_layer, lambda_b, num_inputs, width_hidden_layer, num_layers)
    
    # We now train model
    H0 = obtain_H(model, X_reduced, lambda_w, num_data_points, num_params)
    
    for j in range(num_train_steps):
        train_model_one_step(model, X_train, y_train, epsilon, lambda_w_inputs, lambda_w_hidden_layer, lambda_b)
            
        if j % delay_train_steps == 0:
            H = obtain_H(model, X_reduced, lambda_w, num_data_points, num_params)
            time_evol_NTK[m, j//delay_train_steps] = H[selected_element_NTK] - H0[selected_element_NTK]
            
            
average_time_evol_NTK = np.mean(time_evol_NTK, axis=0)
std_time_evol_NTK = np.std(time_evol_NTK, axis=0) / np.sqrt(num_networks_ensemble)

print(H0[selected_element_NTK])
print(H[selected_element_NTK])

plt.errorbar(list_train_steps, average_time_evol_NTK, yerr=std_time_evol_NTK, fmt='.')
plt.show()

sys.exit()
    
####################################### GRAPHS ##########################################
    
list_colours = ['r', 'g', 'b', 'k']

average_array_mse_H_constant_lambdaw = np.mean(list_array_mse_H_constant_lambdaw, axis=0)
std_array_mse_H_constant_lambdaw = np.sqrt(np.var(list_array_mse_H_constant_lambdaw, axis=0)/repeat_exp)

average_array_mse_H_variable_lambdaw = np.mean(list_array_mse_H_variable_lambdaw, axis=0)
std_array_mse_H_variable_lambdaw = np.sqrt(np.var(list_array_mse_H_variable_lambdaw, axis=0)/repeat_exp)

fig, axs = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle("MSE Error H Matrices")

for i, width_hidden_layer in enumerate(list_width):
    axs[0].errorbar(list_train_steps, average_array_mse_H_constant_lambdaw[i], yerr=std_array_mse_H_constant_lambdaw[i], linestyle='none', marker='.', color=list_colours[i], label=f"Width: {width_hidden_layer}")
    axs[1].errorbar(list_train_steps, average_array_mse_H_variable_lambdaw[i], yerr=std_array_mse_H_variable_lambdaw[i], linestyle='none', marker='x', color=list_colours[i], label=f"Width: {width_hidden_layer}")
    
    axs[0].set_title("Constant $\lambda_w$")
    axs[0].set_xlabel("Training Steps")
    axs[0].set_ylabel("MSE Error H Matrices")
    axs[0].set_yscale('log')
    axs[0].legend()
    
    axs[1].set_title("Variable $\lambda_w$")
    axs[1].set_xlabel("Training Steps")
    axs[1].set_ylabel("MSE Error H Matrices")
    axs[1].set_yscale('log')
    axs[1].legend()

fig.tight_layout()
fig.savefig(f'Data/sidebyside_{repeat_exp}_trials.png', dpi=100, bbox_inches='tight')
plt.show()





fig = plt.figure(figsize=(10, 7))
fig.tight_layout()

for i, width_hidden_layer in enumerate(list_width):
    plt.errorbar(list_train_steps, average_array_mse_H_constant_lambdaw[i], yerr=std_array_mse_H_constant_lambdaw[i], linestyle='none', marker='.', color=list_colours[i])
    plt.errorbar(list_train_steps, average_array_mse_H_variable_lambdaw[i], yerr=std_array_mse_H_variable_lambdaw[i], linestyle='none', marker='x', color=list_colours[i])

plt.xlabel("Training Steps")
plt.ylabel("MSE Error H Matrices")
plt.yscale('log') 

# Create custom legends
import matplotlib.lines as mlines
color_lines = [mlines.Line2D([], [], color=c, marker='.', linestyle='None') for c in list_colours]
regime_lines = [mlines.Line2D([], [], color='black', marker=m, linestyle='None') for m in ['.', 'x']]

# Add legends
list_width_labels = ["Width: " + str(w) for w in list_width]
legend1 = plt.legend(color_lines, list_width_labels, bbox_to_anchor=(1, 0.32), fontsize=12)
plt.gca().add_artist(legend1)
legend2 = plt.legend(regime_lines, ['Constant $\lambda_w$', 'Variable $\lambda_w$'], loc='lower right', fontsize=12)

plt.title("Evolution of MSE Error of H Matrices")
plt.savefig(f'Data/together_{repeat_exp}_trials.png', dpi=100, bbox_inches='tight')
plt.show()

# Save average_array_mse_H_constant_lambdaw to a CSV file
np.savetxt(f'Data/average_array_mse_H_constant_lambdaw_{repeat_exp}_trials.csv', average_array_mse_H_constant_lambdaw, delimiter=',')

# Save average_array_mse_H_variable_lambdaw to a CSV file
np.savetxt(f'Data/average_array_mse_H_variable_lambdaw_{repeat_exp}_trials.csv', average_array_mse_H_variable_lambdaw, delimiter=',')