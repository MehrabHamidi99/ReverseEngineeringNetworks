import pandas as pd
import numpy as np

import torch
import torch.nn as nn
from torch import vmap

from sklearn.linear_model import LinearRegression, RANSACRegressor

import matplotlib.pyplot as plt
import time


from numpy.core.memmap import dtype


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


INPUT_DIM = 2

simple_network = Two_dimensional_input_network(INPUT_DIM, [4, 1])