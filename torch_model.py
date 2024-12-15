import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(101)

class torch_organization_model(nn.Module):
    def __init__(self, input_dim=89, hidden_units=[128, 128], out_dim = 64):
        super(torch_organization_model, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_units[0])
        hidden_layers = []
        for i in range(1,len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_units[-1], out_dim)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x

class torch_top_model(nn.Module):
    def __init__(self, input_dim=89, hidden_units=[128, 128], num_classes=2):
        super(torch_top_model, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_units[0])
        hidden_layers = []
        for i in range(1,len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_units[-1], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        # x = torch.sigmoid(x).squeeze()
        x = self.softmax(x)

        return x

class top_model_for_retraining(nn.Module):
    def __init__(self, base_model):
        super(top_model_for_retraining, self).__init__()
        self.base_model = base_model

    def forward(self, x, client_params):
        # Manually set weights and biases for each layer
        with torch.no_grad():
            self.base_model.input_layer.weight.copy_(client_params['input_layer_weights'])
            self.base_model.input_layer.bias.copy_(client_params['input_layer_biases'])
            self.base_model.output_layer.weight.copy_(client_params['output_layer_weights'])
            self.base_model.output_layer.bias.copy_(client_params['output_layer_biases'])
    
        return self.base_model(x)


class MlpModel(nn.Module):
    def __init__(self, input_dim=89, hidden_units=[128], num_classes=10):
        super(MlpModel, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_units[0])
        hidden_layers = []
        for i in range(1,len(hidden_units)):
            hidden_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            hidden_layers.append(nn.ReLU())
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Linear(hidden_units[-1], num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.softmax(x)
        x = torch.sigmoid(x)

        return x
