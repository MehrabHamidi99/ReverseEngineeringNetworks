import numpy as np
from struct import unpack
from keras.initializers import _compute_fans
from tensorflow.python.ops.random_ops import random_normal
from scipy.ndimage import label
from sklearn.linear_model import RANSACRegressor, LinearRegression
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as a3
import matplotlib.colors as pycolors

MNIST_TRAIN = ('../../datasets/mnist/train-images-idx3-ubyte',
         '../../datasets/mnist/train-labels-idx1-ubyte')
MNIST_TEST = ('../../datasets/mnist/t10k-images-idx3-ubyte',
              '../../datasets/mnist/t10k-labels-idx1-ubyte')

CIFAR_TRAIN = ['../../datasets/cifar-10/data_batch_1',
               '../../datasets/cifar-10/data_batch_2',
               '../../datasets/cifar-10/data_batch_3',
               '../../datasets/cifar-10/data_batch_4',
               '../../datasets/cifar-10/data_batch_5']
CIFAR_TEST = ['../../datasets/cifar-10/test_batch']


def list_to_str(my_list):
    output = ''
    for n, obj in enumerate(my_list):
        if isinstance(obj, int):
            output = output + str(obj)
        else:
            output = output + 'Conv' + str(obj[0])
            if obj[1]:
                output = output + ',Pool'
        if n < len(my_list) - 1:
            output = output + ','
    return output


def count_neurons(network, input_shape):
    if isinstance(network[0], int):
        return np.sum(network)
    else:
        result = 0
        shape = np.array(input_shape)
        for obj in network:
            shape[0:2] -= 2
            shape[2] = obj[0]
            result += np.prod(shape)
            if obj[1]:
                shape[0] = int(shape[0] / 2)
                shape[1] = int(shape[1] / 2)
        return result


def weight_initializer(shape, dtype, partition_info):
    fan_in = _compute_fans(shape)[0]
    return random_normal(shape, stddev=(np.sqrt(2. / fan_in)), dtype=tf.float64)


def bias_initializer(bias_std):
    return lambda shape, dtype, partition_info: random_normal(shape, stddev=bias_std, dtype=tf.float64)


def load_mnist(train):
    if train:
        filenames = MNIST_TRAIN
    else:
        filenames = MNIST_TEST
    with open(filenames[0], "rb") as f:
        _, _, rows, cols = unpack(">IIII", f.read(16))
        X = np.fromfile(f, dtype=np.uint8).reshape(-1, 28, 28, 1) / 255.
    with open(filenames[1], "rb") as f:
        _, _ = unpack(">II", f.read(8))
        Y = np.fromfile(f, dtype=np.int8).reshape(-1)
    X = 2 * X - 1
    return X, Y


def load_cifar(train):
    if train:
        paths = CIFAR_TRAIN
    else:
        paths = CIFAR_TEST
    X = np.zeros((0, 32, 32, 3))
    Y = np.zeros((0,))
    for path in paths:
        with open(path, 'rb') as f:
            dict = pickle.load(f, encoding='bytes')
        X_path = dict[b'data'].reshape(-1, 3, 32, 32).astype(np.float64)
        X_path = np.swapaxes(X_path, 1, 3)
        X_path = np.swapaxes(X_path, 1, 2)
        X_path = X_path / 255.
        X_path = 2 * X_path - 1
        Y_path = np.array(dict[b'labels'])
        X = np.concatenate((X, X_path))
        Y = np.concatenate((Y, Y_path))
    return X, Y


def build_simple_mlp(network, input_shape, output_dim=10, use_output_placeholder_dim=False, activation='relu'):
    input_placeholder = tf.placeholder(tf.float64, (None,) + input_shape, name='X')
    if use_output_placeholder_dim:
        output_placeholder = tf.placeholder(tf.int64, (None, output_dim), name='Y')
    else:
        output_placeholder = tf.placeholder(tf.int64, (None,), name='Y')
    relu = tf.keras.layers.Flatten()(input_placeholder)
    for width in network:
        dense_layer = tf.layers.dense(relu, width, activation=None, use_bias=True,
                                      kernel_initializer=weight_initializer)
        if activation == 'relu':
            relu = tf.nn.relu(dense_layer)
        elif activation == 'tanh':
            relu = tf.nn.tanh(dense_layer)
        else:
            raise NotImplementedError
    output_layer = tf.layers.dense(relu, output_dim, activation=None, use_bias=True,
                                   kernel_initializer=weight_initializer)
    output_layer = tf.identity(output_layer, name='output')
    return input_placeholder, output_placeholder, output_layer


def build_convnet(network, input_shape, output_dim=10, use_output_placeholder_dim=False):
    input_placeholder = tf.placeholder(tf.float64, (None,) + input_shape, name='X')
    if use_output_placeholder_dim:
        output_placeholder = tf.placeholder(tf.int64, (None, output_dim), name='Y')
    else:
        output_placeholder = tf.placeholder(tf.int64, (None,), name='Y')
    relu = input_placeholder
    for (width, pool) in network:
        conv_layer = tf.layers.conv2d(relu, width, (3, 3), kernel_initializer=weight_initializer)
        if pool:
            conv_layer = tf.layers.max_pooling2d(conv_layer, (2, 2), strides=(2, 2))
        relu = tf.nn.relu(conv_layer)
    relu = tf.keras.layers.Flatten()(relu)
    output_layer = tf.layers.dense(relu, output_dim, activation=None, use_bias=True,
                                   kernel_initializer=weight_initializer)
    output_layer = tf.identity(output_layer, name='output')
    return input_placeholder, output_placeholder, output_layer


def random_string():
    return str(np.random.random())[2:]


class LinearRegion1D:
    def __init__(self, param_min, param_max, fn_weight, fn_bias, next_layer_off):
        self._min = param_min
        self._max = param_max
        self._fn_weight = fn_weight
        self._fn_bias = fn_bias
        self._next_layer_off = next_layer_off

    def get_new_regions(self, new_weight_n, new_bias_n, n):
        weight_n = np.dot(self._fn_weight, new_weight_n)
        bias_n = np.dot(self._fn_bias, new_weight_n) + new_bias_n
        if weight_n == 0:
            min_image = bias_n
            max_image = bias_n
        elif weight_n >= 0:
            min_image = weight_n * self._min + bias_n
            max_image = weight_n * self._max + bias_n
        else:
            min_image = weight_n * self._max + bias_n
            max_image = weight_n * self._min + bias_n
        if 0 < min_image:
            return [self]
        elif 0 > max_image:
            self._next_layer_off.append(n)
            return [self]
        else:
            if weight_n == 0:
                return [self]
            else:
                preimage = (-bias_n) / weight_n
                next_layer_off0 = list(np.copy(self._next_layer_off))
                next_layer_off1 = list(np.copy(self._next_layer_off))
                if weight_n >= 0:
                    next_layer_off0.append(n)
                else:
                    next_layer_off1.append(n)
                region0 = LinearRegion1D(self._min, preimage, self._fn_weight, self._fn_bias, next_layer_off0)
                region1 = LinearRegion1D(preimage, self._max, self._fn_weight, self._fn_bias, next_layer_off1)
                return [region0, region1]

    def next_layer(self, new_weight, new_bias):
        self._fn_weight = np.dot(self._fn_weight, new_weight).ravel()
        self._fn_bias = (np.dot(self._fn_bias, new_weight) + new_bias).ravel()
        self._fn_weight[self._next_layer_off] = 0
        self._fn_bias[self._next_layer_off] = 0
        self._next_layer_off = []

    @property
    def max(self):
        return self._max

    @property
    def min(self):
        return self._min

    @property
    def fn_weight(self):
        return self._fn_weight

    @property
    def fn_bias(self):
        return self._fn_bias

    @property
    def next_layer_off(self):
        return self._next_layer_off

    @property
    def dead(self):
        return np.all(np.equal(self._fn_weight, 0))


def regions_1d(the_weights, the_biases, endpt1, endpt2):
    regions = [LinearRegion1D(param_min=0., param_max=1., fn_weight=(endpt2 - endpt1), fn_bias=endpt1,
                              next_layer_off=[])]
    depth = len(the_weights)
    for k in range(depth - 1):
        for n in range(the_biases[k].shape[0]):
            new_regions = []
            for region in regions:
                new_regions = new_regions + region.get_new_regions(the_weights[k][:, n], the_biases[k][n], n)
            regions = new_regions
        for region in regions:
            region.next_layer(the_weights[k], the_biases[k])
    for region in regions:
        region.next_layer(the_weights[-1], the_biases[-1])
    return regions


def region_pts_1d(regions, param_min=-np.inf):
    xs = []
    ys = []
    for region in regions:
        if region.min == param_min:
            pass
        else:
            xs.append(region.min)
            ys.append(region.min * region.fn_weight + region.fn_bias)
    return (xs, ys)


def gradients_1d(regions):
    lengths = []
    gradients = []
    biases = []
    for region in regions:
        lengths.append(region.max - region.min)
        gradients = gradients + list(region.fn_weight)
        biases = biases + list(region.fn_bias)
    return {'lengths': lengths, 'gradients': gradients, 'biases': biases}


def intersect_lines_2d(line_weight, line_bias, pt1, pt2):
    t = (np.dot(pt1, line_weight) + line_bias) / np.dot(pt1 - pt2, line_weight)
    return pt1 + t * (pt2 - pt1)

class LinearRegion2D:
    def __init__(self, fn_weight, fn_bias, vertices, edge_neurons, next_layer_off):
        self._fn_weight = fn_weight
        self._fn_bias = fn_bias
        self._vertices = vertices
        self._edge_neurons = edge_neurons
        self._num_vertices = len(vertices)
        self._next_layer_off = next_layer_off

    def get_new_regions(self, new_weight_n, new_bias_n, n, edge_neuron):
        weight_n = np.dot(self._fn_weight, new_weight_n)
        bias_n = np.dot(self._fn_bias, new_weight_n) + new_bias_n
        vertex_images = np.dot(self._vertices, weight_n) + bias_n
        is_pos = (vertex_images > 0)
        is_neg = np.logical_not(is_pos)  # assumes that distribution of bias_n has no atoms
        if np.all(is_pos):
            return [self]
        elif np.all(is_neg):
            self._next_layer_off.append(n)
            return [self]
        else:
            pos_vertices = []
            neg_vertices = []
            pos_edge_neurons = []
            neg_edge_neurons = []
            for i in range(self._num_vertices):
                j = np.mod(i + 1, self._num_vertices)
                vertex_i = self.vertices[i, :]
                vertex_j = self.vertices[j, :]
                if is_pos[i]:
                    pos_vertices.append(vertex_i)
                    pos_edge_neurons.append(self.edge_neurons[i])
                else:
                    neg_vertices.append(vertex_i)
                    neg_edge_neurons.append(self.edge_neurons[i])
                if is_pos[i] == ~is_pos[j]:
                    intersection = intersect_lines_2d(weight_n, bias_n, vertex_i, vertex_j)
                    pos_vertices.append(intersection)
                    neg_vertices.append(intersection)
                    if is_pos[i]:
                        pos_edge_neurons.append(edge_neuron)
                        neg_edge_neurons.append(self.edge_neurons[i])
                    else:
                        pos_edge_neurons.append(self.edge_neurons[i])
                        neg_edge_neurons.append(edge_neuron)
            pos_vertices = np.array(pos_vertices)
            neg_vertices = np.array(neg_vertices)
            next_layer_off0 = list(np.copy(self._next_layer_off))
            next_layer_off1 = list(np.copy(self._next_layer_off))
            next_layer_off0.append(n)
            region0 = LinearRegion2D(self._fn_weight, self._fn_bias, neg_vertices, neg_edge_neurons, next_layer_off0)
            region1 = LinearRegion2D(self._fn_weight, self._fn_bias, pos_vertices, pos_edge_neurons, next_layer_off1)
            return [region0, region1]

    def next_layer(self, new_weight, new_bias):
        self._fn_weight = np.dot(self._fn_weight, new_weight)
        self._fn_bias = np.dot(self._fn_bias, new_weight) + new_bias
        self._fn_weight[:, self._next_layer_off] = 0
        self._fn_bias[self._next_layer_off] = 0
        self._next_layer_off = []

    @property
    def vertices(self):
        return self._vertices

    @property
    def edge_neurons(self):
        return self._edge_neurons

    @property
    def fn_weight(self):
        return self._fn_weight

    @property
    def fn_bias(self):
        return self._fn_bias

    @property
    def dead(self):
        return np.all(np.equal(self._fn_weight, 0))

def regions_2d(the_weights, the_biases, input_vertices, input_dim=2, seed=0):
    np.random.seed(seed)
    if input_dim == 2:
        input_fn_weight = np.eye(2)
        input_fn_bias = np.zeros((2,))
    else:
        input_fn_weight = 2 * np.random.random((2, input_dim)) - 1
        input_fn_weight[0, :] /= np.linalg.norm(input_fn_weight[0, :])
        input_fn_weight[1, :] /= np.linalg.norm(input_fn_weight[1, :])
        input_fn_bias = np.zeros((input_dim,))
    input_edge_neurons = [() for i in range(input_vertices.shape[0])]
    regions = [LinearRegion2D(input_fn_weight, input_fn_bias, input_vertices, input_edge_neurons, [])]
    depth = len(the_weights)
    for k in range(depth - 1):
        for n in range(the_biases[k].shape[0]):
            new_regions = []
            for region in regions:
                new_regions = new_regions + region.get_new_regions(the_weights[k][:, n], the_biases[k][n], n, (k, n))
            regions = new_regions
        for region in regions:
            region.next_layer(the_weights[k], the_biases[k])
    for region in regions:
        region.next_layer(the_weights[-1], the_biases[-1])
    return regions


def batch_apply(sess, X, eval_size):
    num = X.shape[0]
    if num >= eval_size:
        return sess.run('output:0', feed_dict={'X:0': X, 'Y:0': np.zeros((num,))})[:, 0]
    num_batches = int(np.ceil(num / eval_size))
    results = sess.run('output:0', feed_dict={'X:0': X[:eval_size], 'Y:0': np.zeros((eval_size,))})[:, 0]
    for i in range(1, num_batches):
        start = eval_size * i
        end = start + eval_size
        batch_num = eval_size
        if end > num:
            end = num
            batch_num = end - start
        X_batch = X[start:end, :]
        results = np.hstack((results, sess.run('output:0', feed_dict={'X:0': X_batch,
                                                                      'Y:0': np.zeros((batch_num,))})[:, 0]))
    return results


def approx_1d(sess, endpt1, endpt2, iterations, zero_angle, init_samples=10000, num_used=1):
    endpt1 = endpt1.reshape(1, -1)
    endpt2 = endpt2.reshape(1, -1)
    length = np.linalg.norm(endpt2 - endpt1)
    samples_t = np.arange(0, 1.0000000001, 1./(init_samples + 1), dtype=np.float64)
    for iter in range(iterations):
        num_samples = len(samples_t)
        samples = np.tile(endpt1, (num_samples, 1)) + np.dot(samples_t.reshape(-1, 1), (endpt2 - endpt1))
        preds = sess.run('output:0', feed_dict={'X:0': samples, 'Y:0': np.zeros((num_samples,))})
        if num_used == 1:
            points = np.hstack((preds[:, 0].reshape(-1, 1), samples_t.reshape(-1, 1) / length))
            vecs = points[1:, :] - points[:-1, :]
            unit_vecs = np.divide(vecs, np.tile(np.linalg.norm(vecs, axis=1).reshape(-1, 1), (1, 2)))
            angles = np.arccos(np.sum(np.multiply(unit_vecs[1:], unit_vecs[:-1]), axis=1))
            bent_indices = np.nonzero(angles > zero_angle)[0]
        else:
            raise NotImplementedError
        end_t = samples_t[[0, -1]]
        keep_t = samples_t[bent_indices + 1]
        new_t_1 = (keep_t + samples_t[bent_indices]) / 2
        new_t_2 = (keep_t + samples_t[bent_indices + 2]) / 2
        samples_t = np.sort(np.hstack((end_t, keep_t, new_t_1, new_t_2)))
    return samples_t[1:-1]

def approx_2d_edges(sess, radius, sample_res, eps, eval_size):
    samples = np.meshgrid(np.linspace(-radius, radius, sample_res), np.linspace(-radius, radius, sample_res))
    samples = np.hstack((samples[0].reshape(sample_res ** 2, -1), samples[1].reshape(sample_res ** 2, -1)))
    preds = batch_apply(sess, samples, eval_size)
    preds = preds.reshape((sample_res, sample_res))
    x_diff = np.abs(preds[1:, :] - preds[:-1, :])
    y_diff = np.abs(preds[:, 1:] - preds[:, :-1])
    x_same = np.logical_and(np.greater(x_diff[1:, :], (1 - eps) * x_diff[:-1, :]),
                              np.less(x_diff[1:, :], (1 + eps) * x_diff[:-1, :]))
    y_same = np.logical_and(np.greater(y_diff[:, 1:], (1 - eps) * y_diff[:, :-1]),
                              np.less(y_diff[:, 1:], (1 + eps) * y_diff[:, :-1]))
    edges = np.logical_not(np.logical_and(x_same[:, 1:-1], y_same[1:-1, :]))
    return np.pad(edges, 1, 'constant')

def plot_approx_2d_edges(sess, radius, sample_res, eps, eval_size):
    edges = approx_2d_edges(sess, radius, sample_res, eps, eval_size)
    plt.imshow(edges[::-1, ::])
    plt.xticks([])
    plt.yticks([])
    plt.show()


def calc_1d(sess, endpt1, endpt2):
    weights = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if w.name.endswith('kernel:0')]
    biases = [b for b in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if b.name.endswith('bias:0')]
    [the_weights, the_biases] = sess.run([weights, biases], feed_dict={'X:0': np.zeros((0, len(endpt1))),
                                                                       'Y:0': np.zeros((0,))})
    points = region_pts_1d(regions_1d(the_weights, the_biases, endpt1, endpt2), 0)[0]
    return points


def plot_calc_1d(sess, endpt1, endpt2, ax, output_pts=False):
    assert len(endpt1) == 2, 'plot_calc_1d requires 2D input'
    points = np.array(calc_1d(sess, endpt1, endpt2))
    pts_x = (points * (endpt2[0] - endpt1[0])) + endpt1[0]
    pts_y = (points * (endpt2[1] - endpt1[1])) + endpt1[1]
    ax.scatter(pts_x, pts_y, color='black')
    if output_pts:
        return points


def calc_2d(sess, radius, input_dim=2, seed=0):
    input_vertices = radius * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    weights = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if w.name.endswith('kernel:0')]
    biases = [b for b in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if b.name.endswith('bias:0')]
    [the_weights, the_biases] = sess.run([weights, biases], feed_dict={'X:0': np.zeros((0, input_dim)),
                                                                       'Y:0': np.zeros((0,))})
    regions = regions_2d(the_weights, the_biases, input_vertices, input_dim=input_dim, seed=seed)
    return regions


def plot_calc_2d(regions, ax, seed, edges=False, gradients=False, color_by_layer=True, colors=['blue', 'red', 'gold']):
    np.random.seed(seed)
    min_gradient = np.inf
    max_gradient = -np.inf
    for region in regions:
        gradient = region.fn_weight[0, 0]
        min_gradient = min(min_gradient, gradient)
        max_gradient = max(max_gradient, gradient)
    minimum = np.array([np.inf, np.inf])
    maximum = np.array([-np.inf, -np.inf])
    for region in regions:
        vertices = region.vertices
        minimum = np.minimum(np.min(vertices, axis=0), minimum)
        maximum = np.maximum(np.max(vertices, axis=0), maximum)
        if edges:
            edge_neurons = region.edge_neurons
            num_vertices = vertices.shape[0]
            for i in range(num_vertices):
                np.random.seed(hash(edge_neurons[i]) % (2**20) + seed)
                j = (i + 1) % num_vertices
                if vertices[i, 0] != vertices[j, 0] and vertices[i, 1] != vertices[j, 1]:
                    if color_by_layer:
                        _ = ax.plot([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]],
                                    c=colors[edge_neurons[i][0]])
                    else:
                        _ = ax.plot([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]],
                                    c=np.random.rand(3, 1))
                else:
                    _ = ax.plot([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]], c='black')
            if gradients:
                gradient = region.fn_weight[0, 0]
                gradient = (gradient - min_gradient) / (max_gradient - min_gradient)
                _ = ax.fill(vertices[:, 0], vertices[:, 1], c=np.array([0, gradient, 0]), alpha=0.9)
        else:
            _ = ax.fill(vertices[:, 0], vertices[:, 1], c=np.random.rand(3, 1))
    plt.xticks([], [])
    plt.yticks([], [])
    ax.set_xlim([minimum[0], maximum[0]])
    ax.set_ylim([minimum[1], maximum[1]])
    ax.set_aspect('equal')
    ax.set_xlabel('Input dim 1', size=30)
    ax.set_ylabel('Input dim 2', size=30)


def plot_calc_2d_heights(regions, ax, seed, edges=False, color_by_layer=True, colors=['blue', 'red', 'gold']):
    np.random.seed(seed)
    minimum = np.array([np.inf, np.inf, np.inf])
    maximum = np.array([-np.inf, -np.inf, -np.inf])
    for region in regions:
        vertices = region.vertices
        vertices = np.hstack((vertices, np.dot(vertices, region.fn_weight)[:, 0].reshape(-1, 1) + region.fn_bias[0]))
        minimum = np.minimum(np.min(vertices, axis=0), minimum)
        maximum = np.maximum(np.max(vertices, axis=0), maximum)
        if edges:
            polygon = a3.art3d.Poly3DCollection([vertices])
            polygon.set_color('gray')
            polygon.set_alpha(0.1)
            ax.add_collection3d(polygon)
            edge_neurons = region.edge_neurons
            num_vertices = vertices.shape[0]
            for i in range(num_vertices):
                np.random.seed(hash(edge_neurons[i]) % (2**20) + seed)
                j = (i + 1) % num_vertices
                if vertices[i, 0] != vertices[j, 0] and vertices[i, 1] != vertices[j, 1]:
                    if color_by_layer:
                        _ = ax.plot([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]],
                                    [vertices[i, 2], vertices[j, 2]], c=colors[edge_neurons[i][0]])
                    else:
                        _ = ax.plot([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]],
                                    [vertices[i, 2], vertices[j, 2]], c=np.random.rand(3, 1))
                else:
                    _ = ax.plot([vertices[i, 0], vertices[j, 0]], [vertices[i, 1], vertices[j, 1]],
                                [vertices[i, 2], vertices[j, 2]], c='black')
        else:
            polygon = a3.art3d.Poly3DCollection([vertices])
            polygon.set_color(pycolors.rgb2hex(np.random.random(3)))
            ax.add_collection3d(polygon)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_xlim3d([minimum[0], maximum[0]])
    ax.set_ylim3d([minimum[1], maximum[1]])
    ax.set_zlim3d([minimum[2], maximum[2]])
    ax.set_xlabel('Input dim 1', size=30)
    ax.set_ylabel('Input dim 2', size=30)
    ax.set_zlabel('Function output', size=30)


def pred_1d(sess, endpt1, endpt2, num_samples=1000):
    endpt1 = endpt1.reshape(1, -1)
    endpt2 = endpt2.reshape(1, -1)
    samples_t = np.arange(0, 1.0000000001, 1./(num_samples + 1), dtype=np.float64)
    samples = np.tile(endpt1, (num_samples + 2, 1)) + np.dot(samples_t.reshape(-1, 1), (endpt2 - endpt1))
    return(sess.run('output:0', feed_dict={'X:0': samples, 'Y:0': np.zeros((num_samples + 2,))})[:, 0])


def plot_pred_1d(sess, endpt1, endpt2, num_samples=1000):
    plt.plot(pred_1d(sess, endpt1, endpt2, num_samples=num_samples))
    plt.show()


def count_sides_2d(sess, radius):
    weights = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if w.name.endswith('kernel:0')]
    biases = [b for b in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES) if b.name.endswith('bias:0')]
    [the_weights, the_biases] = sess.run([weights, biases], feed_dict={'X:0': np.zeros((0, 2)), 'Y:0': np.zeros((0,))})
    num_first_layer = the_weights[0].shape[1]
    input_vertices = radius * np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
    regions = regions_2d(the_weights, the_biases, input_vertices)
    results = []
    for i in range(num_first_layer):
        pos_side = 0
        neg_side = 0
        line_weight = the_weights[0][:, i]
        line_bias = the_biases[0][i]
        for region in regions:
            done = False
            j = 0
            while not done:
                vertex = region.vertices[j, :]
                value = np.dot(line_weight, vertex) + line_bias
                if value > 0:
                    pos_side += 1
                    done = True
                elif value < 0:
                    neg_side += 1
                    done = True
                else:
                    j += 1
        results.append((pos_side, neg_side))
    results_array = np.array(results)
    plt.scatter(results_array[:, 0], results_array[:, 1])
    plt.show()
    return results


class Polygon:
    def __init__(self, vertices):
        self._vertices = vertices
        self._num_vertices = vertices.shape[0]

    def line_overlap(self, weight, bias):
        vertex_images = np.dot(self._vertices, weight) + bias
        is_pos = (vertex_images > 0)
        endpt1 = None
        endpt2 = None
        for i in range(self._num_vertices):
            j = np.mod(i + 1, self._num_vertices)
            if is_pos[i] and not is_pos[j]:
                endpt1 = intersect_lines_2d(weight, bias, self.vertices[i, :], self.vertices[j, :])
            elif not is_pos[i] and is_pos[j]:
                endpt2 = intersect_lines_2d(weight, bias, self.vertices[i, :], self.vertices[j, :])
        return endpt1, endpt2

    @property
    def vertices(self):
        return self._vertices


def get_border_pieces(input, border_width=5, threshold=4):
    input_border = np.copy(input)
    input_border[border_width:-border_width, border_width:-border_width] = False
    labeled_array, num_labels = label(input_border)
    output = []
    for i in range(num_labels):
        pixels = np.array(np.nonzero(labeled_array == i))
        if pixels.shape[1] > threshold:
            mean = np.mean(pixels, axis=1)
            output.append(mean)
    return output


def pixels_segment(endpt1, endpt2):
    shifted1 = np.array(endpt1) + 0.5
    shifted2 = np.array(endpt2) + 0.5
    minx, miny = np.minimum(shifted1, shifted2)
    maxx, maxy = np.maximum(shifted1, shifted2)
    crosses1_x = np.arange(np.ceil(minx), np.floor(maxx) + 1)
    if len(crosses1_x) == 0 or minx == maxx:
        crosses1_x = np.array([])
        crosses1_y = np.array([])
    else:
        crosses1_y = ((shifted2[1] - shifted1[1]) / (shifted2[0] - shifted1[0])) * (crosses1_x - shifted1[0]) + shifted1[1]
    crosses2_y = np.arange(np.ceil(miny), np.floor(maxy) + 1)
    if len(crosses2_y) == 0 or miny == maxy:
        crosses2_x = np.array([])
        crosses2_y = np.array([])
    else:
        crosses2_x = ((shifted2[0] - shifted1[0]) / (shifted2[1] - shifted1[1])) * (crosses2_y - shifted1[1]) + shifted1[0]
    if (shifted2[1] - shifted1[1]) * (shifted2[0] - shifted1[0]) > 0:
        results_x = np.hstack((crosses1_x - 1, np.floor(crosses2_x), np.floor(maxx))).astype(int)
        results_y = np.hstack((np.floor(crosses1_y), crosses2_y - 1, np.floor(maxy))).astype(int)
    else:
        results_x = np.hstack((crosses1_x - 1, np.floor(crosses2_x), np.floor(maxx))).astype(int)
        results_y = np.hstack((np.floor(crosses1_y), crosses2_y, np.floor(miny))).astype(int)
    results = (results_x, results_y)
    return results


def fit_line_region(start, input, subset, test_num=501, angle_reduction=5, min_pixels=10):
    assert test_num % 2 == 1, 'test_num must be odd'
    half_num = int((test_num - 1) / 2)
    angle_mean = 0
    angle_radius = np.pi / 2
    prev_frac = 0
    frac = 0
    while prev_frac != frac or frac == 0:
        prev_frac = np.copy(frac)
        test_angles = ((angle_radius / half_num) * np.arange(-half_num, half_num + 1)).reshape(-1, 1) + angle_mean
        test_fracs = []
        for angle in test_angles:
            weight = np.array([np.cos(angle), np.sin(angle)])
            bias = -np.dot(np.array(start), weight)
            endpt1, endpt2 = subset.line_overlap(weight, bias)
            pixels = pixels_segment(endpt1, endpt2)
            if len(pixels[0]) >= min_pixels:
                test_fracs.append(np.mean(input[pixels[0], pixels[1]]))
            else:
                test_fracs.append(0)
        best = np.argmax(test_fracs)
        frac = test_fracs[best]
        angle_mean = test_angles[best]
        angle_radius = angle_radius / angle_reduction
    weight = np.array([np.cos(angle_mean), np.sin(angle_mean)])
    bias = -np.dot(np.array(start), weight)
    endpts = subset.line_overlap(weight, bias)
    return endpts, frac


def approx_1d_2(sess, endpt1, endpt2, iterations, eps, init_samples=1, precision=1e-5, use_outputs=None):
    endpt1 = endpt1.reshape(1, -1)
    endpt2 = endpt2.reshape(1, -1)
    samples_t = np.arange(0, 1.0000000001, 1./(init_samples + 1), dtype=np.float64)
    total_samples = 0
    for iter in range(iterations):
        num_samples = len(samples_t)
        samples = np.tile(endpt1, (num_samples, 1)) + np.dot(samples_t.reshape(-1, 1), (endpt2 - endpt1))
        preds = sess.run('output:0', feed_dict={'X:0': samples, 'Y:0': np.zeros((num_samples,))})
        total_samples += num_samples
        num_outputs = preds.shape[1]
        if use_outputs == None:
            outputs_used = num_outputs
        else:
            outputs_used = use_outputs
        slopes = np.abs(np.divide(preds[1:, 0] - preds[:-1, 0], samples_t[1:] - samples_t[:-1]))
        diff = np.logical_or(np.less(slopes[1:], (1 - eps) * slopes[:-1]),
                             np.greater(slopes[1:], (1 + eps) * slopes[:-1]))
        for i in range(1, outputs_used):
            slopes = np.abs(np.divide(preds[1:, i] - preds[:-1, i], samples_t[1:] - samples_t[:-1]))
            diff = np.logical_or(diff, np.logical_or(np.less(slopes[1:], (1 - eps) * slopes[:-1]),
                                                     np.greater(slopes[1:], (1 + eps) * slopes[:-1])))
        diff_indices = np.nonzero(diff)[0]
        end_t = samples_t[[0, -1]]
        keep_t = samples_t[diff_indices + 1]
        if iter < iterations - 1:
            new_t_1 = (2 * keep_t + samples_t[diff_indices]) / 3
            new_t_2 = (2 * keep_t + samples_t[diff_indices + 2]) / 3
            samples_t = np.sort(np.hstack((end_t, keep_t, new_t_1, new_t_2)))
        else:
            samples_t = keep_t
    if len(samples_t) > 0:
        unique = np.nonzero(samples_t[1:] - samples_t[:-1] > precision)[0]
        output = 0.5 * (samples_t[np.hstack((unique, -1))] + samples_t[np.hstack((0, unique + 1))])
    else:
        output = samples_t
    return output, total_samples


def approx_boundary(sess, point, radius, num_samples, threshold, iterations, eps, on_pos_side_of=None,
                    init_samples=1, precision=1e-4, use_outputs=None, multiple_points = False):
    dim = len(point)
    results = []
    i = 0
    total_samples = 0
    while len(results) < num_samples:
        if len(results) == 0 and i > 10 * num_samples:  # Heuristic stopping point
            print("Exceeded maximum number of samples")
            return None, None, None, None, total_samples
        i += 1
        midpoint = np.random.random((dim,)) - 0.5
        midpoint /= np.linalg.norm(midpoint)
        perp = np.random.random((dim,)) - 0.5
        perp = perp - np.dot(midpoint, perp) * midpoint
        perp /= np.linalg.norm(perp)
        endpt1 = point + radius * midpoint + 5 * radius * perp
        endpt2 = point + radius * midpoint - 5 * radius * perp
        if type(on_pos_side_of) == tuple:
            all_output, samples = approx_1d_2(sess, endpt1, endpt2, iterations, eps, init_samples=init_samples,
                                              precision=precision, use_outputs=use_outputs)
            total_samples += samples
            output = []
            for output_pt in all_output:
                candidate = output_pt * (endpt2 - endpt1) + endpt1
                condition = np.dot(on_pos_side_of[0], candidate - point) > precision
                if condition:
                    output.append(output_pt)
        else:
            output, samples = approx_1d_2(sess, endpt1, endpt2, iterations, eps, init_samples=init_samples,
                                          precision=precision, use_outputs=use_outputs)
            total_samples += samples
        if len(output) == 1:
            results.append(output[0] * (endpt2 - endpt1) + endpt1)
        elif len(output) > 1:
            if not multiple_points:
                return None, None, None, None, total_samples
        else:
            pass
    results = np.array(results)
    X = results[:, :-1]
    y = results[:, -1]
    reg = RANSACRegressor(random_state=0).fit(X, y)
    if reg.score(X, y) < threshold:
        return None, None, None, None, total_samples
    else:
        weight = np.hstack((reg.estimator_.coef_, -1))
        bias = reg.estimator_.intercept_
        bias /= np.linalg.norm(weight)
        weight /= np.linalg.norm(weight)
        if bias < 0:
            bias = -bias
            weight = -weight
        return weight, bias, results[0, :], results, total_samples


def is_straight(sess, point, vec, eps, use_outputs=None):
    samples = np.array([point - vec, point, point + vec])
    preds = sess.run('output:0', feed_dict={'X:0': samples, 'Y:0': np.zeros((3,))})
    num_outputs = preds.shape[1]
    if use_outputs == None:
        outputs_used = num_outputs
    else:
        outputs_used = use_outputs
    straight = True
    slopes1 = []
    slopes2 = []
    for i in range(outputs_used):
        slope1 = preds[1, i] - preds[0, i]
        slope2 = preds[2, i] - preds[1, i]
        slopes1.append(slope1)
        slopes2.append(slope2)
        straight = np.logical_and(straight,
                                  np.logical_and(np.abs(slope2) > (1 - eps) * np.abs(slope1),
                                                 np.abs(slope2) < (1 + eps) * np.abs(slope1)))
    return straight