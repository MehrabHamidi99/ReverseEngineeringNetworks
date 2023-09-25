import pandas as pd
import numpy as np

from networks import *

from torch import vmap

from sklearn.linear_model import LinearRegression, RANSACRegressor

import matplotlib.pyplot as plt
import time


from numpy.core.memmap import dtype

INPUT_DIM = 2

simple_network = Two_dimensional_input_network(INPUT_DIM, [4, 1])

x_s = np.linspace(-5, 5, num=1e6)
y_s = np.linspace(-5, 5, num=1e6)

xv, yv = np.meshgrid(x_s, y_s)

