import torch
import torch.nn as nn
import torch.nn.init as init


class MLP_ReLU(nn.Module):
    def __init__(self, n_in, layer_list):
        super(MLP_ReLU, self).__init__()

        self.input_dim = n_in


        self.first_layer = nn.Sequential(nn.Linear(n_in, layer_list[0]), nn.ReLU())

        self.hidden_layers = nn.Sequential()
        for i in range(1, len(layer_list)):
            self.hidden_layers.add_module(f"linear_{i}", nn.Linear(layer_list[i - 1], layer_list[i]))
            if i != len(layer_list) - 1:
                self.hidden_layers.add_module(f"relu_{i}", nn.ReLU())

        self.apply(self.init_weights)
        self.output_dim = layer_list[-1]

    def forward(self, x):
        first_layer_result = self.first_layer(x)
        output = self.hidden_layers(first_layer_result)
        return output

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def get_all_parameters(self):
        weights = []
        biases = []
        for k, v in self.state_dict().items():
            if 'weight' in k:
                weights.append(v.T)
            if 'bias' in k:
                biases.append(v)
        return weights, biases

class Two_dimensional_input_network(nn.Module):

  Bias_std = 1

  def __init__(self, n_in, layer_list):
    super(MLP_ReLU, self).__init__()

    self.input_dim = n_in

    self.first_layer = nn.Sequential(nn.Linear(n_in, layer_list[0]), nn.ReLU())

    self.hidden_layers = nn.Sequential()
    for i in range(1, len(layer_list)):
      self.hidden_layers.append(nn.Linear(layer_list[i - 1], layer_list[i]))
      if i != len(layer_list) - 1:
        self.hidden_layers.append(nn.ReLU())

    self.apply(self.init_weights)

    self.output_dim = layer_list[-1]

    # self.last_layer = nn.Sequential(nn.Linear(layer_list[-1], n_out), nn.ReLU())
  
  def forward(self, x):
    first_layer_result = self.first_layer(x)
    output = self.hidden_layers(first_layer_result)
    return output

  def initialize_weights(self, layer):
    # Using He-normal and standard normal to initialize weights and biases
    if 'linear' in str(layer.__class__).lower():
      nn.init.kaiming_normal_(layer.weight)
      layer.weight = nn.Parameter(torch.tensor(layer.weight, dtype=torch.float64))
      nn.init.normal_(layer.bias)
      layer.bias = nn.Parameter(torch.tensor(layer.bias, dtype=torch.float64))

  def init_weights(self, m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

  def get_all_parameters(self):
    weights = []
    biases = []
    for k, v in self.state_dict().items():
      if 'weight' in k:
        weights.append(v.T)
      if 'bias' in k:
        biases.append(v)

    return weights, biases

class Simple2DReLU(nn.Module):
    def __init__(self):
        super(Simple2DReLU, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )
        self.layers[0].weight.data = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.layers[0].bias.data = torch.tensor([-2.0, -2.0])
        self.layers[2].weight.data = torch.tensor([[1.0, 1.0]])
        self.layers[2].bias.data = torch.tensor([0.0])

    def forward(self, x):
        return self.layers(x)

class Simple3DReLU(nn.Module):
    def __init__(self):
        super(Simple3DReLU, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3, 3),
            nn.ReLU(),
            nn.Linear(3, 1)
        )
        self.layers[0].weight.data = torch.tensor([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0], [1.0, 0.0, 1.0]])
        self.layers[0].bias.data = torch.tensor([-2.0, -2.0, -2.0])
        self.layers[2].weight.data = torch.tensor([[1.0, 1.0, 1.0]])
        self.layers[2].bias.data = torch.tensor([0.0])

    def forward(self, x):
        return self.layers(x)

class SimpleContinuingNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1)
        )
        # Initialize weights and biases
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.constant_(layer.weight, 1.0)
                init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.layers(x)
    
class BendingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.ReLU(),
            nn.Linear(1, 1)
        )
        # Initialize weights and biases
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.constant_(layer.weight, 1.0)
                init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.layers(x)

class BendingNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 2),
            nn.ReLU(),
            nn.Linear(2, 1),
            nn.ReLU(),
            nn.Linear(1, 1)
        )
        # Initialize weights and biases
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                init.constant_(layer.weight, 1.0)
                init.constant_(layer.bias, 0.0)

    def forward(self, x):
        return self.layers(x)

class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x
