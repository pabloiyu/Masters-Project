import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import time

class NeuralNetwork(nn.Module):
    """
    Defines a simple neural network with a given number of layers and neurons per layer.
    """
    def __init__(self, num_inputs, width_hidden_layer, num_layers=2, use_identity_activation=False):
        super().__init__()
            
        layers = []
        layers.append(nn.Linear(num_inputs,width_hidden_layer))
        if not use_identity_activation:
            layers.append(nn.Tanh())
        
        for _ in range(num_layers-2):
            layers.append(nn.Linear(width_hidden_layer,width_hidden_layer))
            if not use_identity_activation:
                layers.append(nn.Tanh())

        layers.append(nn.Linear(width_hidden_layer,1))
         
        self.linear_tanh_stack = nn.Sequential(*layers) # argument unpacking
        
    def forward(self, x):
        logits = self.linear_tanh_stack(x)
        return logits

def initiate_model(num_inputs, width_hidden_layer, num_layers, Cw=1, use_identity_activation=False, device='cpu'):
    """
    Initializes a neural network model with the given hyperparameters and weights and biases drawn from a normal distribution.
    """
    model = NeuralNetwork(num_inputs, width_hidden_layer, num_layers, use_identity_activation).to(device)

    for layer_num,layer in enumerate(model.linear_tanh_stack):
        if isinstance(layer, nn.Linear):
            layer_weight = layer.weight.data
            num_input_neurons = layer_weight.shape[1]
            num_output_neurons = layer_weight.shape[0]

            variance = Cw / (num_input_neurons)
            nn.init.normal_(layer.weight, mean=0, std=np.sqrt(variance))
            nn.init.zeros_(layer.bias)
            
    return model


def get_lambda_matrix_diagonal(lambda_w_inputs, lambda_w_hidden_layer, lambda_b, num_inputs, width_hidden_layer, num_layers):
    """
    Calculates the diagonal row of the learning rate tensor. Each weight and each bias has its own scaling factor.
    """
    lambda_matrix_diagonal = [lambda_w_inputs]*num_inputs*width_hidden_layer + [lambda_b]*width_hidden_layer
    
    for _ in range(num_layers-2):
        lambda_matrix_diagonal += [lambda_w_hidden_layer]*width_hidden_layer**2 + [lambda_b]*width_hidden_layer
        
    # Final layer
    lambda_matrix_diagonal += [lambda_w_hidden_layer] * width_hidden_layer + [lambda_b]*1
    
    num_params = len(lambda_matrix_diagonal)
    
    return lambda_matrix_diagonal, num_params

def obtain_H(model, X, lambda_matrix_diagonal, num_data_points, num_params):
    """
    Calculates the Neural Tangent Kernel of a given model with respect to the input data X.
    """
    # Initialize a 2D array to store the gradients for all data points
    all_grads = torch.zeros((num_data_points, num_params))
    
    # Loop over all data points
    for i in range(num_data_points):
        # Forward pass
        output = model(X[i])
        
        # Select a specific neuron from the output
        neuron_output = output.squeeze()
        
        # Compute gradients
        neuron_output.backward()
        
        # Store the gradients in the 2D array
        grads = []
        for param in model.parameters():
            # Each of the parameters is either the weight or bias of each layer
            # So for a 2 layer network, we have 4 parameters
            grads.append(param.grad.view(-1).clone())
            param.grad.zero_()
        
        all_grads[i] = torch.cat(grads)
        
    H = torch.einsum('i, ki, li -> kl', torch.tensor(lambda_matrix_diagonal), all_grads, all_grads) #symmetric

    return H
    

def obtain_H_grad(model, X, lambda_matrix_diagonal, num_data_points, num_params):
    """
    Calculates the Neural Tangent Kernel of a given model with respect to the input data X.
    This version explicitly uses the autograd.grad function to compute the gradients.
    """
    params = list(model.parameters())
    all_grads = torch.zeros((num_data_points, num_params))
    
    for i in range(num_data_points):
        jac = torch.autograd.grad(model(X[i].unsqueeze(0)).squeeze(), params, create_graph=True)
        grads = []
        
        for element in jac:
            grads.append(element.view(-1).clone().detach())
            
        grads = torch.cat(grads)
        all_grads[i] = grads
        
    H = np.einsum('i, ki, li -> kl', lambda_matrix_diagonal, all_grads, all_grads) #symmetric

    return H


def obtain_H_jac(model, X, lambda_matrix_diagonal, num_data_points, num_params) -> np.ndarray:
    """
    Compute the Hessian matrix of a given model with respect to the input data X.

    Args:
        model (torch.nn.Module): The model for which to compute the Hessian matrix.
        X (torch.Tensor): The input data.
        lambda_matrix_diagonal (torch.Tensor): The diagonal elements of the lambda matrix.
        num_data_points (int): The number of data points.
        num_params (int): The number of model parameters.

    Returns:
        numpy.ndarray: The Hessian matrix.

    """
    # We use jax-like transforms to perform batch operations such as jacobian calculations.
    params = {k: v.detach() for k, v in model.named_parameters()}

    def fnet_single(params, x):
        return torch.func.functional_call(model, params, (x.unsqueeze(0),)).squeeze(0)
    
    x1 = X
    x2 = X
    
    # Compute J(x1)
    jac1 = torch.func.vmap(torch.func.jacrev(fnet_single), (None, 0))(params, x1)
    jac1 = jac1.values()
    jac1 = [j.flatten(2) for j in jac1]

    # Compute J(x2)
    jac2 = torch.func.vmap(torch.func.jacrev(fnet_single), (None, 0))(params, x2)
    jac2 = jac2.values()
    jac2 = [j.flatten(2) for j in jac2]

    # Compute J(x1) @ J(x2).T
    H = torch.stack([torch.einsum('Naf,Mbf->NMab', j1, j2) for j1, j2 in zip(jac1, jac2)])
    H = H.sum(0)
    H = H.squeeze().numpy()

    return H

def mse_H_matrices(H1,H2):
    return np.mean((H2-H1)**2)

def train_model_one_step(model, X, Y, epsilon, lambda_w_inputs, lambda_w_hidden_layer, lambda_b):
    """
    Performs a single gradient descent step on a batch of data.
    """
    output = model(X)
    loss = F.mse_loss(output.squeeze(), Y)
    
    # Clear previous gradients
    model.zero_grad()
    
    # Compute gradients
    loss.backward()
    
    # Update model parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'bias' in name:
                lambda_factor = lambda_b
            elif 'weight' in name:
                if '0' in name:
                    # If it's the first layer we use the corresponding scaling factor for lambda_w
                    lambda_factor = lambda_w_inputs
                else:
                    lambda_factor = lambda_w_hidden_layer
            else:
                lambda_factor = 0
                print("Wrong Parameter Type")
                
            param -= epsilon * lambda_factor * param.grad if param.grad is not None else 0
            
    # Zero gradients for the next iteration
    model.zero_grad()
    
    
##################################### K #####################################

def obtain_neuron_activation_all_layers_one_input(num_inputs, width_hidden_layer, num_layers, x, Cw=1, use_identity_activation=False, device='cpu'):
    """
    Performs a forward pass with a given input x. It then extracts the intermediate and output activations
    of the model.
    """
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook

    model = NeuralNetwork(num_inputs, width_hidden_layer, num_layers, use_identity_activation).to(device)
    
    for layer_num,layer in enumerate(model.linear_tanh_stack):
        if isinstance(layer, nn.Linear):
            layer_weight = layer.weight.data
            num_input_neurons = layer_weight.shape[1]
            num_output_neurons = layer_weight.shape[0]

            variance = Cw / (num_input_neurons)
            nn.init.normal_(layer.weight, mean=0, std=np.sqrt(variance))
            nn.init.zeros_(layer.bias)
            
            # Forward hook saves intermediate activations of all neurons in the network.
            layer.register_forward_hook(get_features(f"{layer_num}"))
    
    model(x)
    
    ith_activation_different_layers = np.zeros(num_layers)  # (num_layers)
    all_activations_different_layers = []                   # (num_layers, width of each layer)
    
    # We go through the model layers one by one 
    for m,list_all_activations in enumerate(features.values()):
        # Numpy array. Contains the activations of all neurons in the m-th layer.
        list_all_activations = list_all_activations.cpu().numpy()
        
        all_activations_different_layers.append(list_all_activations)
        ith_activation_different_layers[m] = list_all_activations[0]
        
    return ith_activation_different_layers, all_activations_different_layers

def obtain_neuron_activation_all_layers_two_inputs(num_inputs, width_hidden_layer, x1, x2, num_layers, Cw=1, use_identity_activation=False, device='cpu'):
    features = {}
    def get_features(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    model = NeuralNetwork(num_inputs, width_hidden_layer, num_layers, use_identity_activation).to(device)
    
    for layer_num,layer in enumerate(model.linear_tanh_stack):
        if isinstance(layer, nn.Linear):
            layer_weight = layer.weight.data
            num_input_neurons = layer_weight.shape[1]
            num_output_neurons = layer_weight.shape[0]

            variance = Cw / (num_input_neurons)
            nn.init.normal_(layer.weight, mean=0, std=np.sqrt(variance))
            nn.init.zeros_(layer.bias)
            
            layer.register_forward_hook(get_features(f"{layer_num}"))
            
    model(x1)
    
    ith_activation_different_layers_x1 = np.zeros(num_layers) # (num_layers)
    
    # We go through the model layers one by one
    for m,list_all_activations_x1 in enumerate(features.values()):
        list_all_activations_x1 = list_all_activations_x1.cpu().numpy()
        ith_activation_different_layers_x1[m] = list_all_activations_x1[0]
        
    model(x2)
    
    ith_activation_different_layers_x2 = np.zeros(num_layers) # (num_layers)
    
    # We go through the model layers one by one
    for m,list_all_activations_x2 in enumerate(features.values()):
        list_all_activations_x2 = list_all_activations_x2.cpu().numpy()
        ith_activation_different_layers_x2[m] = list_all_activations_x2[0]
    
    return ith_activation_different_layers_x1, ith_activation_different_layers_x2