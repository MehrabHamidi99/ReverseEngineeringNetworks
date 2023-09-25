import numpy as np
from itertools import product
from struct import unpack

from scipy.ndimage import label
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.linear_model import Ridge
from scipy.optimize import minimize

import pickle

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as pycolors
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


import torch
from torch import vmap

def complex_step_gradient(model, input_point, epsilon=1e-20):
    """
    Estimate the gradient of a black-box PyTorch model using complex-step derivative approximation
    and vectorized batch computation with vmap.

    :param model: The black-box PyTorch model (a callable function).
    :param input_point: The input point (NumPy array) at which to compute the gradient.
    :param epsilon: The perturbation value for complex-step approximation.
    :return: The estimated gradient (NumPy array) at the input point.
    """

    # Convert the input_point to a complex NumPy array
    x = input_point.astype(np.complex128)

    # Create a perturbation matrix (each row is a perturbed input) using broadcasting
    perturbation_matrix = np.eye(len(x), dtype=np.complex128) * 1j * epsilon

    # Compute the perturbed inputs
    x_perturbed = x + perturbation_matrix

    # Convert the perturbed inputs to PyTorch tensors
    x_perturbed_torch = torch.tensor(x_perturbed, dtype=torch.cfloat)

    # Apply the model to each perturbed input using vmap
    f_x_perturbed = torch.imag(torch.vmap(model)(x_perturbed_torch)).double().numpy()

    # Compute the complex-step approximation for the gradient
    grad = f_x_perturbed / epsilon

    return grad

def compute_second_order_gradient(model, input_point, epsilon=1e-3):

    # Create a perturbation matrix (each row is a perturbed input)
    perturbation_matrix = np.eye(input_point.size) * epsilon
    x_perturbed_positive = input_point + perturbation_matrix
    x_perturbed_negative = input_point - perturbation_matrix

    # Compute the function values f(X_i) for all perturbed inputs
    # using vmap to batch the computation
    f_prime_x_perturbed_positive = compute_gradient_vmap(model, x_perturbed_positive, epsilon)
    f_prime_x_perturbed_negative = compute_gradient_vmap(model, x_perturbed_negative, epsilon)

    # Compute the central difference approximation for the gradient
    second_order_grad = (f_prime_x_perturbed_positive - f_prime_x_perturbed_negative) / (2 * epsilon)

    return second_order_grad

def compute_partial_derivative(model, input_point, direction, epsilon=1e-3):
    """
    Estimate the partial gradient of a black-box deep ReLU network with respect to a given direction
    """
    # Convert the input_point to a PyTorch tensor
    x_input = torch.from_numpy(input_point).double()
    x_perturb = torch.from_numpy(input_point + epsilon * direction).double()


    f_x = model(x_input).detach().clone()
    f_x_perturbed = model(x_perturb).detach().clone()

    partial_derivative = (f_x_perturbed - f_x) / epsilon

    return partial_derivative.numpy()

def compute_gradient_vmap(model, input_point, epsilon=1e-3):
    """
    Estimate the gradient of a black-box deep ReLU network using central differences and vmap.

    :param model: The black-box PyTorch model (a callable function).
    :param input_point: The input point (NumPy array) at which to compute the gradient.
    :param epsilon: The perturbation value for central differences.
    :return: The estimated gradient (NumPy array) at the input point.
    """

    # Create a perturbation matrix (each row is a perturbed input)
    perturbation_matrix = np.eye(input_point.size) * epsilon
    x_perturbed_positive = torch.from_numpy(input_point + perturbation_matrix).double()
    x_perturbed_negative = torch.from_numpy(input_point - perturbation_matrix).double()

    # Compute the function values f(X_i) for all perturbed inputs
    # using vmap to batch the computation
    f_x_perturbed_positive = torch.vmap(model)(x_perturbed_positive).detach().numpy(dtype=np.double)
    f_x_perturbed_negative = torch.vmap(model)(x_perturbed_negative).detach().numpy(dtype=np.double)

    # Compute the central difference approximation for the gradient
    grad = (f_x_perturbed_positive - f_x_perturbed_negative) / (2 * epsilon)

    return grad

def find_hyperplane_parameters(model, x, epsilon=1e-6):
    """
    Estimate the hyperplane parameters of a deep ReLU network (the black-box model) at a given input point.

    :param model: The black-box model (a callable function).
    :param x: The input point (NumPy array) at which to compute the hyperplane parameters.
    :param epsilon: The epsilon value used for gradient estimation.
    :return: The hyperplane parameters (a tuple with the normal vector and the offset).
    """

    # Compute the gradient at the input point x
    gradient = compute_gradient_vmap(model, x, epsilon)

    # Reshape the gradient to have the same dimensions as the input point
    gradient = gradient.reshape(x.shape)

    # Compute the function value at the input point x
    f_x = model(torch.tensor(x, dtype=torch.float32)).item()

    # The gradient is the normal vector of the hyperplane
    normal_vector = gradient

    # The offset can be computed as the dot product between the normal vector and the input point minus the function value
    offset = -np.dot(normal_vector, x) + f_x

    return normal_vector, offset


def find_hyperplane_parameters_regression_prior(model, x, n_samples=50, radius=1e-3, alpha=0.0):
    """
    Estimate the hyperplane parameters of a deep ReLU network (the black-box model) at a given input point using
    ridge regression with a prior on the gradient.

    :param model: The black-box model (a callable function).
    :param x: The input point (NumPy array) at which to compute the hyperplane parameters.
    :param n_samples: The number of samples to be drawn around the input point.
    :param radius: The radius of the sphere around the input point from which samples are drawn.
    :param alpha: The regularization strength for ridge regression.
    :return: The estimated hyperplane parameters (a tuple with the normal vector and the offset).
    """

    # Sample points around x using a Gaussian distribution
    samples = np.random.normal(loc=x, scale=radius, size=(n_samples, x.shape[0]))

    # Compute the function values at the sampled points
    f_samples = np.array([model(torch.tensor(sample, dtype=torch.float32)).item() for sample in samples])

    # Subtract the input point from the samples
    # X = samples - x

    # Perform ridge regression with prior on the gradient
    clf = LinearRegression(alpha=alpha)
    clf.fit(samples, f_samples)

    # Get the estimated normal vector and offset from the ridge regression model
    normal_vector = clf.coef_
    offset = clf.intercept_

    return normal_vector, offset

def find_hyperplane_parameters_regression(model, x, n_points=50, radius=1e-2):
    # Sample points along each axis
    dim = x.shape[0]
    points = np.zeros((dim, n_points, dim))
    for i in range(dim):
        perturbations = np.random.uniform(-radius, radius, n_points)
        points[i, :, i] = perturbations

    # Combine the points from all axes
    points = points.reshape(-1, dim)
    points += x

    # Evaluate the model on the sampled points
    y = np.array([model(torch.tensor(point, dtype=torch.float32)).item() for point in points])

    # Perform linear regression to estimate the hyperplane parameters
    reg = LinearRegression().fit(points, y)

    # Obtain the coefficients and intercept
    normal_vector = reg.coef_
    offset = reg.intercept_

    return normal_vector, offset

def generate_orthogonal_vectors(normal_vector):
    """
    Generate n-1 orthogonal vectors on an n-1 dimensional hyperplane in n-dimensional space.

    :param normal_vector: The normal vector of the hyperplane (NumPy array).
    :return: A NumPy array of shape (n-1, n) containing the orthogonal vectors.
    """

    n = len(normal_vector)
    random_vectors = np.random.randn(n - 1, n)

    # Normalize normal_vector
    normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)

    # Subtract the projection of the normalized normal_vector from each random vector
    random_vectors -= (random_vectors @ normal_vector_normalized).reshape(-1, 1) * normal_vector_normalized

    # Normalize each vector
    random_vectors /= np.linalg.norm(random_vectors, axis=1).reshape(-1, 1)

    # Vectorized Gram-Schmidt process
    Q, _ = np.linalg.qr(random_vectors.T)
    orthogonal_vectors = Q.T

    return orthogonal_vectors

def find_intersection_on_hyperplane(model, start_point, direction_vector, left, right, n_dim, epsilon=1e-6, num_steps=None):
    """
    Find the intersection point of a deep ReLU network's hyperplanes by moving over a line on the current hyperplane.
    This function is a recursive binary search on the line.

    :param model: The black-box model (a callable function).
    :param start_point: The starting point (NumPy array) in the input space.
    :param direction_vector: The direction vector (NumPy array) on the hyperplane.
    :param left: The left endpoint of the search interval.
    :param right: The right endpoint of the search interval.
    :param n_dim: The number of dimensions of the input space.
    :param epsilon: The epsilon value used for gradient estimation.
    :param num_steps: The number of steps to perform binary search on the line.
    :return: The intersection point (NumPy array) of the hyperplanes.
    """

    # Base case: if the number of steps is 0 or None, return the current intersection point
    if num_steps == 0 or num_steps is None:
        return start_point + right * direction_vector

    # Find the hyperplane parameters at the starting point
    normal_vector, _ = find_hyperplane_parameters_regression(model, start_point, epsilon)

    # Calculate the mid-point of the search interval
    mid = (left + right) / 2

    # Move to the mid-point along the direction vector
    mid_point = start_point + mid * direction_vector

    # Compute the gradient at the mid-point
    grad_mid = compute_gradient_vmap(model, mid_point, epsilon)

    # Check if the gradient changes
    if not np.allclose(grad_mid, normal_vector, atol=epsilon) or np.abs(np.linalg.norm(grad_mid) - np.linalg.norm(normal_vector)) > epsilon:
        # If the gradient changes, there is an intersection point in the current search interval
        return find_intersection_on_hyperplane(model, start_point, direction_vector, left, mid, n_dim, epsilon, num_steps - 1)
    else:
        # If the gradient does not change, there is no intersection point in the current search interval
        return find_intersection_on_hyperplane(model, start_point, direction_vector, mid, right, n_dim, epsilon, num_steps - 1)

def does_hyperplane_bend(model, intersection_point, normal_vector, offset, epsilon=1e-6, threshold=1e-3):
    """
    Determine if the given hyperplane will bend at the intersection point.

    :param model: The black-box model (a callable function).
    :param intersection_point: The intersection point (NumPy array).
    :param normal_vector: The normal vector of the hyperplane (NumPy array).
    :param offset: The offset of the hyperplane.
    :param epsilon: The epsilon value used for gradient estimation.
    :param threshold: The threshold to compare gradients.
    :return: A boolean value indicating whether the hyperplane bends at the intersection point.
    """

    # Compute the gradient at the intersection point
    gradient_at_intersection = compute_gradient_vmap(model, intersection_point, epsilon)

    # Normalize the gradient and the normal vector
    gradient_at_intersection_normalized = gradient_at_intersection / np.linalg.norm(gradient_at_intersection)
    normal_vector_normalized = normal_vector / np.linalg.norm(normal_vector)

    print(normal_vector_normalized)
    print(gradient_at_intersection_normalized)

    # Compare the normalized gradient with the normalized normal vector
    is_close = np.allclose(gradient_at_intersection_normalized, normal_vector_normalized, atol=threshold)

    # If the normalized gradient and the normalized normal vector are not close,
    # it means the hyperplane bends at the intersection point
    return not is_close

def is_function_differentiable(model, x, epsilon=1e-6):
    """
    Check if a piecewise linear function (deep ReLU network) is differentiable at a given point x.

    :param model: The black-box model (a callable function).
    :param x: The input point (NumPy array) at which to check differentiability.
    :param epsilon: The perturbation value for checking differentiability.
    :return: A boolean indicating if the function is differentiable at the input point.
    """

    n_dim = x.size

    # Iterate over all dimensions
    for dim in range(n_dim):
        # Perturb the input point in the positive and negative directions in the current dimension
        x_positive = x.copy()
        x_negative = x.copy()
        x_positive[dim] += epsilon
        x_negative[dim] -= epsilon

        # Compute gradients at both perturbed points
        grad_x_positive = compute_gradient_vmap(model, x_positive, epsilon)
        grad_x_negative = compute_gradient_vmap(model, x_negative, epsilon)

        # Check if the gradient changes between the two points
        if not np.allclose(grad_x_positive, grad_x_negative, rtol=1e-4, atol=1e-4):
            return False

    return True

def find_2d_boundaries(model, x_range, y_range, num_points_per_dim, epsilon=1e-5):
    boundaries = []

    # Generate the grid points in the x and y ranges
    x_points = np.linspace(x_range[0], x_range[1], num_points_per_dim)
    y_points = np.linspace(y_range[0], y_range[1], num_points_per_dim)

    # Iterate over all combinations of x and y points
    for x, y in tqdm.tqdm(product(x_points, y_points)):
        point = np.array([x, y])

        # Compute the gradient at the current point
        grad = compute_gradient_vmap(model, point, epsilon)

        # Check if the gradient changes between the previous and current point
        if len(boundaries) > 0:
            grad_prev = boundaries[-1][1]
            if not np.allclose(np.linalg.norm(grad), np.linalg.norm(grad_prev), atol=1e-1):
                # Store the point and its gradient
                boundaries.append((point, grad))

        # If this is the first point, store the point and its gradient
        else:
            boundaries.append((point, grad))

    return boundaries

def find_2d_boundaries_v3(model, x_range, y_range, num_points_per_dim, epsilon=1e-5):
    # Generate the grid points in the x and y ranges
    x_points = np.linspace(x_range[0], x_range[1], num_points_per_dim)
    y_points = np.linspace(y_range[0], y_range[1], num_points_per_dim)

    # Create a 2D grid of points
    X, Y = np.meshgrid(x_points, y_points)
    grid_points = np.stack((X, Y), axis=-1)

    # Compute the gradient at each grid point
    grad = np.array([compute_gradient_vmap(model, point, epsilon) for point in grid_points.reshape(-1, 2)]).reshape(grid_points.shape)

    # Calculate the gradient norms
    grad_norms = np.linalg.norm(grad, axis=2)

    # Find the differences between adjacent gradient norms
    diff_x = np.abs(np.diff(grad_norms, axis=0))
    diff_y = np.abs(np.diff(grad_norms, axis=1))

    # Compute the boundary mask
    boundary_mask_x = diff_x > 5e-2
    boundary_mask_y = diff_y > 5e-2

    # Find the boundary points
    boundary_points_x = grid_points[:-1][boundary_mask_x]
    boundary_points_y = grid_points[:, :-1][boundary_mask_y]

    # Concatenate the boundary points from both dimensions
    boundary_points = np.concatenate((boundary_points_x, boundary_points_y), axis=0)

    return boundary_points

def traverse_cube_edges(model, n_dim, cube_length, step_size, epsilon, grad_threshold, similarity_threshold):
    edge_points = np.arange(-cube_length / 2, cube_length / 2 + step_size, step_size)
    edge_hyperplanes = defaultdict(list)

    # Iterate over all dimensions
    for dim in range(n_dim):
        # Iterate over all pairs of edge points
        for idx1, p1 in enumerate(edge_points[:-1]):
            p2 = edge_points[idx1 + 1]

            # Create points along the edge in the current dimension
            x1 = np.zeros(n_dim)
            x2 = np.zeros(n_dim)
            x1[dim] = p1
            x2[dim] = p2

            # Compute gradients at both points
            grad_x1 = compute_gradient_vmap(model, x1, epsilon)
            grad_x2 = compute_gradient_vmap(model, x2, epsilon)

            # Check if the gradient changes between the two points
            if not (np.allclose(np.abs(grad_x1), np.abs(grad_x2), atol=grad_threshold * n_dim * 10) and
                    np.allclose(np.linalg.norm(grad_x1), np.linalg.norm(grad_x2), atol=grad_threshold)):
                # Find the hyperplane parameters
                normal_vector, offset = find_hyperplane_parameters(model, x1, epsilon)

                # Check if the current normal vector is similar to any of the previously found normal vectors
                similar_hyperplane_exists = False
                for existing_normal, _ in edge_hyperplanes[dim]:
                    if np.allclose(normal_vector, existing_normal, atol=similarity_threshold * n_dim * 10) and \
                       np.allclose(np.linalg.norm(normal_vector), np.linalg.norm(existing_normal), atol=similarity_threshold):
                        similar_hyperplane_exists = True
                        break


                # If the normal vector is unique, store the hyperplane parameters
                if not similar_hyperplane_exists:
                    edge_hyperplanes[dim].append((normal_vector, offset))

    return edge_hyperplanes

def traverse_cube_edges_v2(model, n_dim, cube_length, step_size, epsilon, grad_threshold=1e-3):
    edge_points = np.arange(0, cube_length + step_size, step_size)
    edge_hyperplanes = defaultdict(list)

    # Iterate over all dimensions
    for dim in range(n_dim):
        # Iterate over all pairs of edge points
        for idx1, p1 in enumerate(edge_points[:-1]):
            p2 = edge_points[idx1 + 1]

            # Create points along the edge in the current dimension
            x1 = np.zeros(n_dim)
            x2 = np.zeros(n_dim)
            x1[dim] = p1
            x2[dim] = p2

            # Compute gradients at both points
            grad_x1 = compute_gradient_vmap(model, x1, epsilon)
            grad_x2 = compute_gradient_vmap(model, x2, epsilon)

            # Check if the gradient changes between the two points, considering the grad_threshold
            if not np.allclose(np.abs(grad_x1), np.abs(grad_x2), atol=grad_threshold):
                # Find the hyperplane parameters
                normal_vector, offset = find_hyperplane_parameters(model, x1, epsilon)

                # Store the hyperplane parameters
                edge_hyperplanes[dim].append((normal_vector, offset))

    return edge_hyperplanes

# You can use this function to find similar hyperplanes
def find_similar_hyperplanes(edge_hyperplanes, similarity_threshold):
    similar_hyperplanes = []

    for dim, hyperplanes in edge_hyperplanes.items():
        for i, (normal_vector1, offset1) in enumerate(hyperplanes):
            for j, (normal_vector2, offset2) in enumerate(hyperplanes[i+1:]):
                if np.allclose(normal_vector1, normal_vector2, atol=similarity_threshold) and np.allclose(offset1, offset2, atol=similarity_threshold):
                    similar_hyperplanes.append((normal_vector1, offset1))

    return similar_hyperplanes

def visualize_surface(model, x_range=(-5, 5), y_range=(-5, 5), num_points=500):
    x_min, x_max = x_range
    y_min, y_max = y_range

    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)

    X, Y = np.meshgrid(x, y)
    input_points = np.array([X.flatten(), Y.flatten()]).T
    input_points_tensor = torch.tensor(input_points, dtype=torch.float32)

    with torch.no_grad():
        Z = model(input_points_tensor).numpy()

    Z = Z.reshape(X.shape)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

def visualize_boundaries(network, x_range=(-5, 5), y_range=(-5, 5), num_points=100):
    x_min, x_max = x_range
    y_min, y_max = y_range

    x_vals = np.linspace(x_min, x_max, num_points)
    y_vals = np.linspace(y_min, y_max, num_points)

    X, Y = np.meshgrid(x_vals, y_vals)

    Z_1 = np.zeros((num_points, num_points))
    Z_2 = np.zeros((num_points, num_points))

    with torch.no_grad():
        for i in range(num_points):
            for j in range(num_points):
                input_data = torch.tensor([X[i, j], Y[i, j]], dtype=torch.float32).unsqueeze(0)
                first_layer_output = network.first_layer(input_data)
                second_layer_input = network.hidden_layers[0](first_layer_output)
                Z_1[i, j] = torch.sum(torch.abs(second_layer_input)).item()

                second_layer_output = network.hidden_layers[1](second_layer_input)
                third_layer_input = network.hidden_layers[2](second_layer_output)
                Z_2[i, j] = torch.sum(torch.abs(third_layer_input)).item()

    fig, ax = plt.subplots()
    ax.contour(X, Y, Z_1, levels=[0.5], colors='blue')
    ax.contour(X, Y, Z_2, levels=[0.5], colors='red')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()

def generate_points(n_points, n_dim, lower_bound, upper_bound):
    points = np.random.uniform(lower_bound, upper_bound, size=(n_points, n_dim))
    return points

def estimate_hyperplane(point, model_function, epsilon=1e-5, n_samples=50):
    dim = len(point)
    perturbations = np.random.randn(n_samples, dim) * epsilon
    perturbed_points = point + perturbations

    # Query the model_function at perturbed points
    perturbed_values = np.array([model_function(x) for x in perturbed_points])

    # Fit a linear regression model
    lr = LinearRegression(fit_intercept=True)
    lr.fit(perturbed_points, perturbed_values)

    # Extract the estimated normal vector and offset
    normal_vector = lr.coef_
    offset = lr.intercept_

    return normal_vector, offset

def calculate_radii(points, hyperplanes, model_function, epsilon=1e-5, n_samples=50):
    n_points = len(points)
    radii = []

    for i in range(n_points):
        point_i = points[i]
        normal_vector_i, offset_i = hyperplanes[i]

        for j in range(n_points):
            if i == j:
                continue

            point_j = points[j]
            normal_vector_j, offset_j = hyperplanes[j]

            # Calculate the distance between hyperplanes
            d = np.abs(offset_i - offset_j) / np.linalg.norm(normal_vector_i - normal_vector_j)

            # Calculate the bound that guarantees convexity
            bound = 0.5 * d

            # Define g_i_prime and g_j_prime within the bound
            g_i_prime = lambda x: np.dot(normal_vector_i, x - point_i) + offset_i if np.linalg.norm(x - point_i) <= bound else 0
            g_j_prime = lambda x: np.dot(normal_vector_j, x - point_j) + offset_j if np.linalg.norm(x - point_j) <= bound else 0

            # Calculate the difference between g_i_prime and g_j_prime
            diff_g = lambda x: g_i_prime(x) - g_j_prime(x)

            # Estimate the maximum difference between g_i_prime and g_j_prime within the bound
            perturbations = np.random.randn(n_samples, len(point_i)) * bound
            perturbed_points = point_i + perturbations
            max_diff = np.max(np.abs([diff_g(x) for x in perturbed_points]))

            # Add the radius to the list
            radii.append(max_diff)

    return radii

# Define the objective function L(h)
def objective_function(weights, points, radii, hyperplanes, model_function):
    lh = 0
    for x in points:
        model_val = model_function(torch.tensor(x, dtype=torch.float32)).detach().numpy()
        weighted_sum = sum([weights[i] * (np.dot(hyperplanes[i][0], x - points[i]) + hyperplanes[i][1]) if np.linalg.norm(x - points[i]) <= radii[i] else 0 for i in range(len(points))])
        lh += (model_val - weighted_sum) ** 2
    return lh

# Define the gradient of the objective function
def gradient_objective_function(weights, points, radii, hyperplanes, model_function):
    grad = np.zeros_like(weights)
    for i in range(len(points)):
        point_i = points[i]
        normal_vector_i, offset_i = hyperplanes[i]
        g_i_prime = lambda x: np.dot(normal_vector_i, x - point_i) + offset_i if np.linalg.norm(x - point_i) <= radii[i] else 0
        for x in points:
            model_val = model_function(torch.tensor(x, dtype=torch.float32)).detach().numpy()
            weighted_sum = sum([weights[j] * (np.dot(hyperplanes[j][0], x - points[j]) + hyperplanes[j][1]) if np.linalg.norm(x - points[j]) <= radii[j] else 0 for j in range(len(points))])
            grad[i] += 2 * (model_val - weighted_sum) * (-g_i_prime(x))
    return grad

# Gradient descent algorithm
def gradient_descent(points, radii, hyperplanes, model_function, initial_weights, learning_rate, num_iterations):
    weights = initial_weights
    losses = []

    for _ in tqdm.tqdm(range(num_iterations)):
        grad = gradient_objective_function(weights, points, radii, hyperplanes, model_function)
        weights -= learning_rate * grad
        loss = objective_function(weights, points, radii, hyperplanes, model_function)
        losses.append(loss)

    return weights, losses

def objective_function_with_lasso(weights, points, hyperplanes, radii, model_function, lambda_lasso):
    h = lambda x: sum([weights[k] * (np.dot(hyperplanes[k][0], x - points[k]) + hyperplanes[k][1]) if np.linalg.norm(x - points[k]) <= radii[k] else 0 for k in range(len(points))])
    objective = np.mean([(h(x) - model_function(x))**2 for x in points])
    lasso_term = lambda_lasso * np.sum(np.abs(weights))
    return objective + lasso_term

def gradient_with_lasso(weights, points, hyperplanes, radii, model_function, lambda_lasso):
    N = len(points)
    gradient_obj = np.zeros(N)

    for k in range(N):
        for i in range(N):
            x = points[i]
            h_k = np.dot(hyperplanes[k][0], x - points[k]) + hyperplanes[k][1] if np.linalg.norm(x - points[k]) <= radii[k] else 0
            h = sum([weights[j] * (np.dot(hyperplanes[j][0], x - points[j]) + hyperplanes[j][1]) if np.linalg.norm(x - points[j]) <= radii[j] else 0 for j in range(len(points))])
            gradient_obj[k] += 2 * (h - model_function(torch.tensor(x, dtype=torch.float32)).detach().numpy()) * h_k
        gradient_obj[k] /= N

    gradient_lasso = lambda_lasso * np.sign(weights)
    gradient = gradient_obj + gradient_lasso

    return gradient

def gradient_with_l2(weights, points, hyperplanes, radii, model_function, lambda_l2):
    N = len(points)
    gradient_obj = np.zeros(N)

    for k in range(N):
        for i in range(N):
            x = points[i]
            h_k = np.dot(hyperplanes[k][0], x - points[k]) + hyperplanes[k][1] if np.linalg.norm(x - points[k]) <= radii[k] else 0
            h = sum([weights[j] * (np.dot(hyperplanes[j][0], x - points[j]) + hyperplanes[j][1]) if np.linalg.norm(x - points[j]) <= radii[j] else 0 for j in range(len(points))])
            gradient_obj[k] += 2 * (h - model_function(torch.tensor(x, dtype=torch.float32)).detach().numpy()) * h_k
        gradient_obj[k] /= N

    gradient_l2 = 2 * lambda_l2 * weights
    gradient = gradient_obj + gradient_l2

    return gradient

def gradient_descent_l2_loss(weights, points, hyperplanes, radii, model_function, lambda_l2, learning_rate, num_iterations):
    for i in tqdm.tqdm(range(num_iterations)):
        gradient = gradient_with_l2(weights, points, hyperplanes, radii, model_function, lambda_l2)
        weights -= learning_rate * gradient

    return weights

def calculate_accuracy(weights, hyperplanes, points, radii, validation_data):
    total_points = len(validation_data)
    correct_points = 0

    for x, true_value in validation_data:
        approx_value = sum([weights[k] * (np.dot(hyperplanes[k][0], x - points[k]) + hyperplanes[k][1]) if np.linalg.norm(x - points[k]) <= radii[k] else 0 for k in range(len(points))])
        if np.isclose(approx_value, true_value, rtol=1e-5, atol=1e-5):
            correct_points += 1

    return correct_points / total_points

def l1_regularized_objective_function(weights, hyperplanes, points, radii, lambd):
    # Compute the integral of the squared difference between the function and the weighted sum of hyperplanes
    integral = 0
    for x in np.nditer(points):
        h_sum = sum([weights[k] * (np.dot(hyperplanes[k][0], x - points[k]) + hyperplanes[k][1]) if np.linalg.norm(x - points[k]) <= radii[k] else 0 for k in range(len(points))])
        integral += (h_sum - model_function(x))**2
    integral *= np.product(np.diff(points, axis=0))

    # Add the L1 regularization term
    l1_regularization = lambd * np.sum(np.abs(weights))

    return integral + l1_regularization





################################################################


# MNIST_TRAIN = ('../../datasets/mnist/train-images-idx3-ubyte',
#          '../../datasets/mnist/train-labels-idx1-ubyte')
# MNIST_TEST = ('../../datasets/mnist/t10k-images-idx3-ubyte',
#               '../../datasets/mnist/t10k-labels-idx1-ubyte')

# CIFAR_TRAIN = ['../../datasets/cifar-10/data_batch_1',
#                '../../datasets/cifar-10/data_batch_2',
#                '../../datasets/cifar-10/data_batch_3',
#                '../../datasets/cifar-10/data_batch_4',
#                '../../datasets/cifar-10/data_batch_5']
# CIFAR_TEST = ['../../datasets/cifar-10/test_batch']


# def list_to_str(my_list):
#     output = ''
#     for n, obj in enumerate(my_list):
#         if isinstance(obj, int):
#             output = output + str(obj)
#         else:
#             output = output + 'Conv' + str(obj[0])
#             if obj[1]:
#                 output = output + ',Pool'
#         if n < len(my_list) - 1:
#             output = output + ','
#     return output


# def count_neurons(network, input_shape):
#     if isinstance(network[0], int):
#         return np.sum(network)
#     else:
#         result = 0
#         shape = np.array(input_shape)
#         for obj in network:
#             shape[0:2] -= 2
#             shape[2] = obj[0]
#             result += np.prod(shape)
#             if obj[1]:
#                 shape[0] = int(shape[0] / 2)
#                 shape[1] = int(shape[1] / 2)
#         return result


# def weight_initializer(shape, dtype, partition_info):
#     fan_in = _compute_fans(shape)[0]
#     return random_normal(shape, stddev=(np.sqrt(2. / fan_in)), dtype=tf.float64)


# def bias_initializer(bias_std):
#     return lambda shape, dtype, partition_info: random_normal(shape, stddev=bias_std, dtype=tf.float64)


# def load_mnist(train):
#     if train:
#         filenames = MNIST_TRAIN
#     else:
#         filenames = MNIST_TEST
#     with open(filenames[0], "rb") as f:
#         _, _, rows, cols = unpack(">IIII", f.read(16))
#         X = np.fromfile(f, dtype=np.uint8).reshape(-1, 28, 28, 1) / 255.
#     with open(filenames[1], "rb") as f:
#         _, _ = unpack(">II", f.read(8))
#         Y = np.fromfile(f, dtype=np.int8).reshape(-1)
#     X = 2 * X - 1
#     return X, Y


# def load_cifar(train):
#     if train:
#         paths = CIFAR_TRAIN
#     else:
#         paths = CIFAR_TEST
#     X = np.zeros((0, 32, 32, 3))
#     Y = np.zeros((0,))
#     for path in paths:
#         with open(path, 'rb') as f:
#             dict = pickle.load(f, encoding='bytes')
#         X_path = dict[b'data'].reshape(-1, 3, 32, 32).astype(np.float64)
#         X_path = np.swapaxes(X_path, 1, 3)
#         X_path = np.swapaxes(X_path, 1, 2)
#         X_path = X_path / 255.
#         X_path = 2 * X_path - 1
#         Y_path = np.array(dict[b'labels'])
#         X = np.concatenate((X, X_path))
#         Y = np.concatenate((Y, Y_path))
#     return X, Y


# def build_simple_mlp(network, input_shape, output_dim=10, use_output_placeholder_dim=False, activation='relu'):
#     input_placeholder = tf.placeholder(tf.float64, (None,) + input_shape, name='X')
#     if use_output_placeholder_dim:
#         output_placeholder = tf.placeholder(tf.int64, (None, output_dim), name='Y')
#     else:
#         output_placeholder = tf.placeholder(tf.int64, (None,), name='Y')
#     relu = tf.keras.layers.Flatten()(input_placeholder)
#     for width in network:
#         dense_layer = tf.layers.dense(relu, width, activation=None, use_bias=True,
#                                       kernel_initializer=weight_initializer)
#         if activation == 'relu':
#             relu = tf.nn.relu(dense_layer)
#         elif activation == 'tanh':
#             relu = tf.nn.tanh(dense_layer)
#         else:
#             raise NotImplementedError
#     output_layer = tf.layers.dense(relu, output_dim, activation=None, use_bias=True,
#                                    kernel_initializer=weight_initializer)
#     output_layer = tf.identity(output_layer, name='output')
#     return input_placeholder, output_placeholder, output_layer


# def build_convnet(network, input_shape, output_dim=10, use_output_placeholder_dim=False):
#     input_placeholder = tf.placeholder(tf.float64, (None,) + input_shape, name='X')
#     if use_output_placeholder_dim:
#         output_placeholder = tf.placeholder(tf.int64, (None, output_dim), name='Y')
#     else:
#         output_placeholder = tf.placeholder(tf.int64, (None,), name='Y')
#     relu = input_placeholder
#     for (width, pool) in network:
#         conv_layer = tf.layers.conv2d(relu, width, (3, 3), kernel_initializer=weight_initializer)
#         if pool:
#             conv_layer = tf.layers.max_pooling2d(conv_layer, (2, 2), strides=(2, 2))
#         relu = tf.nn.relu(conv_layer)
#     relu = tf.keras.layers.Flatten()(relu)
#     output_layer = tf.layers.dense(relu, output_dim, activation=None, use_bias=True,
#                                    kernel_initializer=weight_initializer)
#     output_layer = tf.identity(output_layer, name='output')
#     return input_placeholder, output_placeholder, output_layer


# def random_string():
#     return str(np.random.random())[2:]


# class LinearRegion1D:
#     def __init__(self, param_min, param_max, fn_weight, fn_bias, next_layer_off):
#         self._min = param_min
#         self._max = param_max
#         self._fn_weight = fn_weight
#         self._fn_bias = fn_bias
#         self._next_layer_off = next_layer_off

#     def get_new_regions(self, new_weight_n, new_bias_n, n):
#         weight_n = np.dot(self._fn_weight, new_weight_n)
#         bias_n = np.dot(self._fn_bias, new_weight_n) + new_bias_n
#         if weight_n == 0:
#             min_image = bias_n
#             max_image = bias_n
#         elif weight_n >= 0:
#             min_image = weight_n * self._min + bias_n
#             max_image = weight_n * self._max + bias_n
#         else:
#             min_image = weight_n * self._max + bias_n
#             max_image = weight_n * self._min + bias_n
#         if 0 < min_image:
#             return [self]
#         elif 0 > max_image:
#             self._next_layer_off.append(n)
#             return [self]
#         else:
#             if weight_n == 0:
#                 return [self]
#             else:
#                 preimage = (-bias_n) / weight_n
#                 next_layer_off0 = list(np.copy(self._next_layer_off))
#                 next_layer_off1 = list(np.copy(self._next_layer_off))
#                 if weight_n >= 0:
#                     next_layer_off0.append(n)
#                 else:
#                     next_layer_off1.append(n)
#                 region0 = LinearRegion1D(self._min, preimage, self._fn_weight, self._fn_bias, next_layer_off0)
#                 region1 = LinearRegion1D(preimage, self._max, self._fn_weight, self._fn_bias, next_layer_off1)
#                 return [region0, region1]

#     def next_layer(self, new_weight, new_bias):
#         self._fn_weight = np.dot(self._fn_weight, new_weight).ravel()
#         self._fn_bias = (np.dot(self._fn_bias, new_weight) + new_bias).ravel()
#         self._fn_weight[self._next_layer_off] = 0
#         self._fn_bias[self._next_layer_off] = 0
#         self._next_layer_off = []

#     @property
#     def max(self):
#         return self._max

#     @property
#     def min(self):
#         return self._min

#     @property
#     def fn_weight(self):
#         return self._fn_weight

#     @property
#     def fn_bias(self):
#         return self._fn_bias

#     @property
#     def next_layer_off(self):
#         return self._next_layer_off

#     @property
#     def dead(self):
#         return np.all(np.equal(self._fn_weight, 0))


# def regions_1d(the_weights, the_biases, endpt1, endpt2):
#     regions = [LinearRegion1D(param_min=0., param_max=1., fn_weight=(endpt2 - endpt1), fn_bias=endpt1,
#                               next_layer_off=[])]
#     depth = len(the_weights)
#     for k in range(depth - 1):
#         for n in range(the_biases[k].shape[0]):
#             new_regions = []
#             for region in regions:
#                 new_regions = new_regions + region.get_new_regions(the_weights[k][:, n], the_biases[k][n], n)
#             regions = new_regions
#         for region in regions:
            # region.next_layer(the_weights[k], the_biases[k])
#     for region in regions:
#         region.next_layer(the_weights[-1], the_biases[-1])
#     return regions


# def region_pts_1d(regions, param_min=-np.inf):
#     xs = []
#     ys = []
#     for region in regions:
#         if region.min == param_min:
#             pass
#         else:
#             xs.append(region.min)
#             ys.append(region.min * region.fn_weight + region.fn_bias)
#     return (xs, ys)


# def gradients_1d(regions):
#     lengths = []
#     gradients = []
#     biases = []
#     for region in regions:
#         lengths.append(region.max - region.min)
#         gradients = gradients + list(region.fn_weight)
#         biases = biases + list(region.fn_bias)
#     return {'lengths': lengths, 'gradients': gradients, 'biases': biases}


# def intersect_lines_2d(line_weight, line_bias, pt1, pt2):
#     t = (np.dot(pt1, line_weight) + line_bias) / np.dot(pt1 - pt2, line_weight)
#     return pt1 + t * (pt2 - pt1)

# class LinearRegion2D:
#     def __init__(self, fn_weight, fn_bias, vertices, edge_neurons, next_layer_off):
#         self._fn_weight = fn_weight
#         self._fn_bias = fn_bias
#         self._vertices = vertices
#         self._edge_neurons = edge_neurons
#         self._num_vertices = len(vertices)
#         self._next_layer_off = next_layer_off

#     def get_new_regions(self, new_weight_n, new_bias_n, n, edge_neuron):
#         weight_n = np.dot(self._fn_weight, new_weight_n)
#         bias_n = np.dot(self._fn_bias, new_weight_n) + new_bias_n
#         vertex_images = np.dot(self._vertices, weight_n) + bias_n
#         is_pos = (vertex_images > 0)
#         is_neg = np.logical_not(is_pos)  # assumes that distribution of bias_n has no atoms
#         if np.all(is_pos):
#             return [self]
#         elif np.all(is_neg):
#             self._next_layer_off.append(n)
#             return [self]
#         else:
#             pos_vertices = []
#             neg_vertices = []
#             pos_edge_neurons = []
#             neg_edge_neurons = []
#             for i in range(self._num_vertices):
#                 j = np.mod(i + 1, self._num_vertices)
#                 vertex_i = self.vertices[i, :]
#                 vertex_j = self.vertices[j, :]
#                 if is_pos[i]:
#                     pos_vertices.append(vertex_i)
#                     pos_edge_neurons.append(self.edge_neurons[i])
#                 else:
#                     neg_vertices.append(vertex_i)
#                     neg_edge_neurons.append(self.edge_neurons[i])
#                 if is_pos[i] == ~is_pos[j]:
#                     intersection = intersect_lines_2d(weight_n, bias_n, vertex_i, vertex_j)
#                     pos_vertices.append(intersection)
#                     neg_vertices.append(intersection)
#                     if is_pos[i]:
#                         pos_edge_neurons.append(edge_neuron)
#                         neg_edge_neurons.append(self.edge_neurons[i])
#                     else:
#                         pos_edge_neurons.append(self.edge_neurons[i])
#                         neg_edge_neurons.append(edge_neuron)
#             pos_vertices = np.array(pos_vertices)
#             neg_vertices = np.array(neg_vertices)
#             next_layer_off0 = list(np.copy(self._next_layer_off))
#             next_layer_off1 = list(np.copy(self._next_layer_off))
#             next_layer_off0.append(n)
#             region0 = LinearRegion2D(self._fn_weight, self._fn_bias, neg_vertices, neg_edge_neurons, next_layer_off0)
#             region1 = LinearRegion2D(self._fn_weight, self._fn_bias, pos_vertices, pos_edge_neurons, next_layer_off1)
#             return [region0, region1]

#     def next_layer(self, new_weight, new_bias):
#         self._fn_weight = np.dot(self._fn_weight, new_weight)
#         self._fn_bias = np.dot(self._fn_bias, new_weight) + new_bias
#         self._fn_weight[:, self._next_layer_off] = 0
#         self._fn_bias[self._next_layer_off] = 0
#         self._next_layer_off = []

#     @property
#     def vertices(self):
#         return self._vertices

#     @property
#     def edge_neurons(self):
#         return self._edge_neurons

#     @property
#     def fn_weight(self):
#         return self._fn_weight

#     @property
#     def fn_bias(self):
#         return self._fn_bias

#     @property
#     def dead(self):
#         return np.all(np.equal(self._fn_weight, 0))

# def regions_2d(the_weights, the_biases, input_vertices, input_dim=2, seed=0):
#     np.random.seed(seed)
#     if input_dim == 2:
#         input_fn_weight = np.eye(2)
#         input_fn_bias = np.zeros((2,))
#     else:
#         input_fn_weight = 2 * np.random.random((2, input_dim)) - 1
#         input_fn_weight[0, :] /= np.linalg.norm(input_fn_weight[0, :])
#         input_fn_weight[1, :] /= np.linalg.norm(input_fn_weight[1, :])
#         input_fn_bias = np.zeros((input_dim,))
#     input_edge_neurons = [() for i in range(input_vertices.shape[0])]
#     regions = [LinearRegion2D(input_fn_weight, input_fn_bias, input_vertices, input_edge_neurons, [])]
#     depth = len(the_weights)
#     for k in range(depth - 1):
#         for n in range(the_biases[k].shape[0]):
#             new_regions = []
#             for region in regions:
#                 new_regions = new_regions + region.get_new_regions(the_weights[k][:, n], the_biases[k][n], n, (k, n))
#             regions = new_regions
#         for region in regions:
#             region.next_layer(the_weights[k], the_biases[k])
#     for region in regions:
#         region.next_layer(the_weights[-1], the_biases[-1])
#     return regions


# def batch_apply(sess, X, eval_size):
#     num = X.shape[0]
#     if num >= eval_size:
#         return sess.run('output:0', feed_dict={'X:0': X, 'Y:0': np.zeros((num,))})[:, 0]
#     num_batches = int(np.ceil(num / eval_size))
#     results = sess.run('output:0', feed_dict={'X:0': X[:eval_size], 'Y:0': np.zeros((eval_size,))})[:, 0]
#     for i in range(1, num_batches):
#         start = eval_size * i
#         end = start + eval_size
#         batch_num = eval_size
#         if end > num:
#             end = num
#             batch_num = end - start
#         X_batch = X[start:end, :]
#         results = np.hstack((results, sess.run('output:0', feed_dict={'X:0': X_batch,
#                                                                       'Y:0': np.zeros((batch_num,))})[:, 0]))
#     return results


# def approx_1d(sess, endpt1, endpt2, iterations, zero_angle, init_samples=10000, num_used=1):
#     endpt1 = endpt1.reshape(1, -1)
#     endpt2 = endpt2.reshape(1, -1)
#     length = np.linalg.norm(endpt2 - endpt1)
#     samples_t = np.arange(0, 1.0000000001, 1./(init_samples + 1), dtype=np.float64)
#     for iter in range(iterations):
#         num_samples = len(samples_t)
#         samples = np.tile(endpt1, (num_samples, 1)) + np.dot(samples_t.reshape(-1, 1), (endpt2 - endpt1))
#         preds = sess.run('output:0', feed_dict={'X:0': samples, 'Y:0': np.zeros((num_samples,))})
#         if num_used == 1:
#             points = np.hstack((preds[:, 0].reshape(-1, 1), samples_t.reshape(-1, 1) / length))
#             vecs = points[1:, :] - points[:-1, :]
#             unit_vecs = np.divide(vecs, np.tile(np.linalg.norm(vecs, axis=1).reshape(-1, 1), (1, 2)))
#             angles = np.arccos(np.sum(np.multiply(unit_vecs[1:], unit_vecs[:-1]), axis=1))
#             bent_indices = np.nonzero(angles > zero_angle)[0]
#         else:
#             raise NotImplementedError
#         end_t = samples_t[[0, -1]]
#         keep_t = samples_t[bent_indices + 1]
#         new_t_1 = (keep_t + samples_t[bent_indices]) / 2
#         new_t_2 = (keep_t + samples_t[bent_indices + 2]) / 2
#         samples_t = np.sort(np.hstack((end_t, keep_t, new_t_1, new_t_2)))
#     return samples_t[1:-1]

# def approx_2d_edges(sess, radius, sample_res, eps, eval_size):
#     samples = np.meshgrid(np.linspace(-radius, radius, sample_res), np.linspace(-radius, radius, sample_res))
#     samples = np.hstack((samples[0].reshape(sample_res ** 2, -1), samples[1].reshape(sample_res ** 2, -1)))
#     preds = batch_apply(sess, samples, eval_size)
#     preds = preds.reshape((sample_res, sample_res))
#     x_diff = np.abs(preds[1:, :] - preds[:-1, :])
#     y_diff = np.abs(preds[:, 1:] - preds[:, :-1])
#     x_same = np.logical_and(np.greater(x_diff[1:, :], (1 - eps) * x_diff[:-1, :]),
#                               np.less(x_diff[1:, :], (1 + eps) * x_diff[:-1, :]))
#     y_same = np.logical_and(np.greater(y_diff[:, 1:], (1 - eps) * y_diff[:, :-1]),
#                               np.less(y_diff[:, 1:], (1 + eps) * y_diff[:, :-1]))
#     edges = np.logical_not(np.logical_and(x_same[:, 1:-1], y_same[1:-1, :]))
#     return np.pad(edges, 1, 'constant')

# def plot_approx_2d_edges(sess, radius, sample_res, eps, eval_size):
#     edges = approx_2d_edges(sess, radius, sample_res, eps, eval_size)
#     plt.imshow(edges[::-1, ::])
#     plt.xticks([])
#     plt.yticks([])
#     plt.show()


# def calc_1d(sess, endpt1, endpt2):
#     weights = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if w.name.endswith('kernel:0')]
#     biases = [b for b in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if b.name.endswith('bias:0')]
#     [the_weights, the_biases] = sess.run([weights, biases], feed_dict={'X:0': np.zeros((0, len(endpt1))),
#                                                                        'Y:0': np.zeros((0,))})
#     points = region_pts_1d(regions_1d(the_weights, the_biases, endpt1, endpt2), 0)[0]
#     return points


# def plot_calc_1d(sess, endpt1, endpt2, ax, output_pts=False):
#     assert len(endpt1) == 2, 'plot_calc_1d requires 2D input'
#     points = np.array(calc_1d(sess, endpt1, endpt2))
#     pts_x = (points * (endpt2[0] - endpt1[0])) + endpt1[0]
#     pts_y = (points * (endpt2[1] - endpt1[1])) + endpt1[1]
#     ax.scatter(pts_x, pts_y, color='black')
#     if output_pts:
#         return points


# def calc_2d(sess, radius, input_dim=2, seed=0):
#     input_vertices = radius * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
#     weights = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if w.name.endswith('kernel:0')]
#     biases = [b for b in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if b.name.endswith('bias:0')]
#     [the_weights, the_biases] = sess.run([weights, biases], feed_dict={'X:0': np.zeros((0, input_dim)),
#                                                                        'Y:0': np.zeros((0,))})
#     regions = regions_2d(the_weights, the_biases, input_vertices, input_dim=input_dim, seed=seed)
#     return regions


# def plot_calc_2d(regions, ax, seed, edges=False, gradients=False, color_by_layer=True, colors=['blue', 'red', 'gold']):
#     np.random.seed(seed)
#     min_gradient = np.inf
#     max_gradient = -np.inf
#     for region in regions:
#         gradient = region.fn_weight[0, 0]
#         min_gradient = min(min_gradient, gradient)
#         max_gradient = max(max_gradient, gradient)
#     minimum = np.array([np.inf, np.inf])
#     maximum = np.array([-np.inf, -np.inf])
#     for region in regions:
#         vertices = region.vertices
#         minimum = np.minimum(np.min(vertices, axis=0), minimum)
#         maximum = np.maximum(np.max(vertices, axis=0), maximum)
#         if edges:
#             edge_neurons = region.edge_neurons
#             num_vertices = vertices.shape[0]
#             for i in range(num_vertices):
#                 np.random.seed(hash(edge_neurons[i]) % (2**20) + seed)
#                 j = (i + 1) % num_vertices
#                 if vertices[i, 0] != vertices[j, 0] and vertices[i, 1] != vertices[j, 1]:
#                     if color_by_layer:
#                         _ = ax.plot([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]],
#                                     c=colors[edge_neurons[i][0]])
#                     else:
#                         _ = ax.plot([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]],
#                                     c=np.random.rand(3, 1))
#                 else:
#                     _ = ax.plot([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]], c='black')
#             if gradients:
#                 gradient = region.fn_weight[0, 0]
#                 gradient = (gradient - min_gradient) / (max_gradient - min_gradient)
#                 _ = ax.fill(vertices[:, 0], vertices[:, 1], c=np.array([0, gradient, 0]), alpha=0.9)
#         else:
#             _ = ax.fill(vertices[:, 0], vertices[:, 1], c=np.random.rand(3, 1))
#     plt.xticks([], [])
#     plt.yticks([], [])
#     ax.set_xlim([minimum[0], maximum[0]])
#     ax.set_ylim([minimum[1], maximum[1]])
#     ax.set_aspect('equal')
#     ax.set_xlabel('Input dim 1', size=30)
#     ax.set_ylabel('Input dim 2', size=30)


# def plot_calc_2d_heights(regions, ax, seed, edges=False, color_by_layer=True, colors=['blue', 'red', 'gold']):
#     np.random.seed(seed)
#     minimum = np.array([np.inf, np.inf, np.inf])
#     maximum = np.array([-np.inf, -np.inf, -np.inf])
#     for region in regions:
#         vertices = region.vertices
#         vertices = np.hstack((vertices, np.dot(vertices, region.fn_weight)[:, 0].reshape(-1, 1) + region.fn_bias[0]))
#         minimum = np.minimum(np.min(vertices, axis=0), minimum)
#         maximum = np.maximum(np.max(vertices, axis=0), maximum)
#         if edges:
#             polygon = a3.art3d.Poly3DCollection([vertices])
#             polygon.set_color('gray')
#             polygon.set_alpha(0.1)
#             ax.add_collection3d(polygon)
#             edge_neurons = region.edge_neurons
#             num_vertices = vertices.shape[0]
#             for i in range(num_vertices):
#                 np.random.seed(hash(edge_neurons[i]) % (2**20) + seed)
#                 j = (i + 1) % num_vertices
#                 if vertices[i, 0] != vertices[j, 0] and vertices[i, 1] != vertices[j, 1]:
#                     if color_by_layer:
#                         _ = ax.plot([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]],
#                                     [vertices[i, 2], vertices[j, 2]], c=colors[edge_neurons[i][0]])
#                     else:
#                         _ = ax.plot([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]],
#                                     [vertices[i, 2], vertices[j, 2]], c=np.random.rand(3, 1))
#                 else:
#                     _ = ax.plot([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]],
#                                 [vertices[i, 2], vertices[j, 2]], c='black')
#         else:
#             polygon = a3.art3d.Poly3DCollection([vertices])
#             polygon.set_color(pycolors.rgb2hex(np.random.random(3)))
#             ax.add_collection3d(polygon)
#     ax.set_aspect('equal')
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])
#     ax.set_xlim3d([minimum[0], maximum[0]])
#     ax.set_ylim3d([minimum[1], maximum[1]])
#     ax.set_zlim3d([minimum[2], maximum[2]])
#     ax.set_xlabel('Input dim 1', size=30)
#     ax.set_ylabel('Input dim 2', size=30)
#     ax.set_zlabel('Function output', size=30)


# def pred_1d(sess, endpt1, endpt2, num_samples=1000):
#     endpt1 = endpt1.reshape(1, -1)
#     endpt2 = endpt2.reshape(1, -1)
#     samples_t = np.arange(0, 1.0000000001, 1./(num_samples + 1), dtype=np.float64)
#     samples = np.tile(endpt1, (num_samples + 2, 1)) + np.dot(samples_t.reshape(-1, 1), (endpt2 - endpt1))
#     return(sess.run('output:0', feed_dict={'X:0': samples, 'Y:0': np.zeros((num_samples + 2,))})[:, 0])


# def plot_pred_1d(sess, endpt1, endpt2, num_samples=1000):
#     plt.plot(pred_1d(sess, endpt1, endpt2, num_samples=num_samples))
#     plt.show()


# def count_sides_2d(sess, radius):
#     weights = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if w.name.endswith('kernel:0')]
#     biases = [b for b in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if b.name.endswith('bias:0')]
#     [the_weights, the_biases] = sess.run([weights, biases], feed_dict={'X:0': np.zeros((0, 2)), 'Y:0': np.zeros((0,))})
#     num_first_layer = the_weights[0].shape[1]
#     input_vertices = radius * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
#     regions = regions_2d(the_weights, the_biases, input_vertices)
#     results = []
#     for i in range(num_first_layer):
#         pos_side = 0
#         neg_side = 0
#         line_weight = the_weights[0][:, i]
#         line_bias = the_biases[0][i]
#         for region in regions:
#             done = False
#             j = 0
#             while not done:
#                 vertex = region.vertices[j, :]
#                 value = np.dot(line_weight, vertex) + line_bias
#                 if value > 0:
#                     pos_side += 1
#                     done = True
#                 elif value < 0:
#                     neg_side += 1
#                     done = True
#                 else:
#                     j += 1
#         results.append((pos_side, neg_side))
#     results_array = np.array(results)
#     plt.scatter(results_array[:, 0], results_array[:, 1])
#     plt.show()
#     return results


# class Polygon:
#     def __init__(self, vertices):
#         self._vertices = vertices
#         self._num_vertices = vertices.shape[0]

#     def line_overlap(self, weight, bias):
#         vertex_images = np.dot(self._vertices, weight) + bias
#         is_pos = (vertex_images > 0)
#         endpt1 = None
#         endpt2 = None
#         for i in range(self._num_vertices):
#             j = np.mod(i + 1, self._num_vertices)
#             if is_pos[i] and not is_pos[j]:
#                 endpt1 = intersect_lines_2d(weight, bias, self.vertices[i, :], self.vertices[j, :])
#             elif not is_pos[i] and is_pos[j]:
#                 endpt2 = intersect_lines_2d(weight, bias, self.vertices[i, :], self.vertices[j, :])
#         return endpt1, endpt2

#     @property
#     def vertices(self):
#         return self._vertices


# def get_border_pieces(input, border_width=5, threshold=4):
#     input_border = np.copy(input)
#     input_border[border_width:-border_width, border_width:-border_width] = False
#     labeled_array, num_labels = label(input_border)
#     output = []
#     for i in range(num_labels):
#         pixels = np.array(np.nonzero(labeled_array == i))
#         if pixels.shape[1] > threshold:
#             mean = np.mean(pixels, axis=1)
#             output.append(mean)
#     return output


# def pixels_segment(endpt1, endpt2):
#     shifted1 = np.array(endpt1) + 0.5
#     shifted2 = np.array(endpt2) + 0.5
#     minx, miny = np.minimum(shifted1, shifted2)
#     maxx, maxy = np.maximum(shifted1, shifted2)
#     crosses1_x = np.arange(np.ceil(minx), np.floor(maxx) + 1)
#     if len(crosses1_x) == 0 or minx == maxx:
#         crosses1_x = np.array([])
#         crosses1_y = np.array([])
#     else:
#         crosses1_y = ((shifted2[1] - shifted1[1]) / (shifted2[0] - shifted1[0])) * (crosses1_x - shifted1[0]) + shifted1[1]
#     crosses2_y = np.arange(np.ceil(miny), np.floor(maxy) + 1)
#     if len(crosses2_y) == 0 or miny == maxy:
#         crosses2_x = np.array([])
#         crosses2_y = np.array([])
#     else:
#         crosses2_x = ((shifted2[0] - shifted1[0]) / (shifted2[1] - shifted1[1])) * (crosses2_y - shifted1[1]) + shifted1[0]
#     if (shifted2[1] - shifted1[1]) * (shifted2[0] - shifted1[0]) > 0:
#         results_x = np.hstack((crosses1_x - 1, np.floor(crosses2_x), np.floor(maxx))).astype(int)
#         results_y = np.hstack((np.floor(crosses1_y), crosses2_y - 1, np.floor(maxy))).astype(int)
#     else:
#         results_x = np.hstack((crosses1_x - 1, np.floor(crosses2_x), np.floor(maxx))).astype(int)
#         results_y = np.hstack((np.floor(crosses1_y), crosses2_y, np.floor(miny))).astype(int)
#     results = (results_x, results_y)
#     return results


# def fit_line_region(start, input, subset, test_num=501, angle_reduction=5, min_pixels=10):
#     assert test_num % 2 == 1, 'test_num must be odd'
#     half_num = int((test_num - 1) / 2)
#     angle_mean = 0
#     angle_radius = np.pi / 2
#     prev_frac = 0
#     frac = 0
#     while prev_frac != frac or frac == 0:
#         prev_frac = np.copy(frac)
#         test_angles = ((angle_radius / half_num) * np.arange(-half_num, half_num + 1)).reshape(-1, 1) + angle_mean
#         test_fracs = []
#         for angle in test_angles:
#             weight = np.array([np.cos(angle), np.sin(angle)])
#             bias = -np.dot(np.array(start), weight)
#             endpt1, endpt2 = subset.line_overlap(weight, bias)
#             pixels = pixels_segment(endpt1, endpt2)
#             if len(pixels[0]) >= min_pixels:
#                 test_fracs.append(np.mean(input[pixels[0], pixels[1]]))
#             else:
#                 test_fracs.append(0)
#         best = np.argmax(test_fracs)
#         frac = test_fracs[best]
#         angle_mean = test_angles[best]
#         angle_radius = angle_radius / angle_reduction
#     weight = np.array([np.cos(angle_mean), np.sin(angle_mean)])
#     bias = -np.dot(np.array(start), weight)
#     endpts = subset.line_overlap(weight, bias)
#     return endpts, frac


# def approx_1d_2(sess, endpt1, endpt2, iterations, eps, init_samples=1, precision=1e-5, use_outputs=None):
#     endpt1 = endpt1.reshape(1, -1)
#     endpt2 = endpt2.reshape(1, -1)
#     samples_t = np.arange(0, 1.0000000001, 1./(init_samples + 1), dtype=np.float64)
#     total_samples = 0
#     for iter in range(iterations):
#         num_samples = len(samples_t)
#         samples = np.tile(endpt1, (num_samples, 1)) + np.dot(samples_t.reshape(-1, 1), (endpt2 - endpt1))
#         preds = sess.run('output:0', feed_dict={'X:0': samples, 'Y:0': np.zeros((num_samples,))})
#         total_samples += num_samples
#         num_outputs = preds.shape[1]
#         if use_outputs == None:
#             outputs_used = num_outputs
#         else:
#             outputs_used = use_outputs
#         slopes = np.abs(np.divide(preds[1:, 0] - preds[:-1, 0], samples_t[1:] - samples_t[:-1]))
#         diff = np.logical_or(np.less(slopes[1:], (1 - eps) * slopes[:-1]),
#                              np.greater(slopes[1:], (1 + eps) * slopes[:-1]))
#         for i in range(1, outputs_used):
#             slopes = np.abs(np.divide(preds[1:, i] - preds[:-1, i], samples_t[1:] - samples_t[:-1]))
#             diff = np.logical_or(diff, np.logical_or(np.less(slopes[1:], (1 - eps) * slopes[:-1]),
#                                                      np.greater(slopes[1:], (1 + eps) * slopes[:-1])))
#         diff_indices = np.nonzero(diff)[0]
#         end_t = samples_t[[0, -1]]
#         keep_t = samples_t[diff_indices + 1]
#         if iter < iterations - 1:
#             new_t_1 = (2 * keep_t + samples_t[diff_indices]) / 3
#             new_t_2 = (2 * keep_t + samples_t[diff_indices + 2]) / 3
#             samples_t = np.sort(np.hstack((end_t, keep_t, new_t_1, new_t_2)))
#         else:
#             samples_t = keep_t
#     if len(samples_t) > 0:
#         unique = np.nonzero(samples_t[1:] - samples_t[:-1] > precision)[0]
#         output = 0.5 * (samples_t[np.hstack((unique, -1))] + samples_t[np.hstack((0, unique + 1))])
#     else:
#         output = samples_t
#     return output, total_samples


# def approx_boundary(sess, point, radius, num_samples, threshold, iterations, eps, on_pos_side_of=None,
#                     init_samples=1, precision=1e-4, use_outputs=None, multiple_points = False):
#     dim = len(point)
#     results = []
#     i = 0
#     total_samples = 0
#     while len(results) < num_samples:
#         if len(results) == 0 and i > 10 * num_samples:  # Heuristic stopping point
#             print("Exceeded maximum number of samples")
#             return None, None, None, None, total_samples
#         i += 1
#         midpoint = np.random.random((dim,)) - 0.5
#         midpoint /= np.linalg.norm(midpoint)
#         perp = np.random.random((dim,)) - 0.5
#         perp = perp - np.dot(midpoint, perp) * midpoint
#         perp /= np.linalg.norm(perp)
#         endpt1 = point + radius * midpoint + 5 * radius * perp
#         endpt2 = point + radius * midpoint - 5 * radius * perp
#         if type(on_pos_side_of) == tuple:
#             all_output, samples = approx_1d_2(sess, endpt1, endpt2, iterations, eps, init_samples=init_samples,
#                                               precision=precision, use_outputs=use_outputs)
#             total_samples += samples
#             output = []
#             for output_pt in all_output:
#                 candidate = output_pt * (endpt2 - endpt1) + endpt1
#                 condition = np.dot(on_pos_side_of[0], candidate - point) > precision
#                 if condition:
#                     output.append(output_pt)
#         else:
#             output, samples = approx_1d_2(sess, endpt1, endpt2, iterations, eps, init_samples=init_samples,
#                                           precision=precision, use_outputs=use_outputs)
#             total_samples += samples
#         if len(output) == 1:
#             results.append(output[0] * (endpt2 - endpt1) + endpt1)
#         elif len(output) > 1:
#             if not multiple_points:
#                 return None, None, None, None, total_samples
#         else:
#             pass
#     results = np.array(results)
#     X = results[:, :-1]
#     y = results[:, -1]
#     reg = RANSACRegressor(random_state=0).fit(X, y)
#     if reg.score(X, y) < threshold:
#         return None, None, None, None, total_samples
#     else:
#         weight = np.hstack((reg.estimator_.coef_, -1))
#         bias = reg.estimator_.intercept_
#         bias /= np.linalg.norm(weight)
#         weight /= np.linalg.norm(weight)
#         if bias < 0:
#             bias = -bias
#             weight = -weight
#         return weight, bias, results[0, :], results, total_samples


# def is_straight(sess, point, vec, eps, use_outputs=None):
#     samples = np.array([point - vec, point, point + vec])
#     preds = sess.run('output:0', feed_dict={'X:0': samples, 'Y:0': np.zeros((3,))})
#     num_outputs = preds.shape[1]
#     if use_outputs == None:
#         outputs_used = num_outputs
#     else:
#         outputs_used = use_outputs
#     straight = True
#     slopes1 = []
#     slopes2 = []
#     for i in range(outputs_used):
#         slope1 = preds[1, i] - preds[0, i]
#         slope2 = preds[2, i] - preds[1, i]
#         slopes1.append(slope1)
#         slopes2.append(slope2)
#         straight = np.logical_and(straight,
#                                   np.logical_and(np.abs(slope2) > (1 - eps) * np.abs(slope1),
#                                                  np.abs(slope2) < (1 + eps) * np.abs(slope1)))
#     return straight