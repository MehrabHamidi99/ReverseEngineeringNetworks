import numpy as np
import tensorflow as tf
from time import time
import os
import pickle
from sklearn.linear_model import RANSACRegressor
from utils import calc_1d, approx_1d_2, approx_boundary, is_straight, list_to_str, plot_calc_2d
from zlib import adler32

NETWORKS = [[10, 10, 10], [20, 10, 10], [30, 10, 10], [40, 10, 10], [50, 10, 10],
            [10, 10, 10, 10], [20, 10, 10, 10], [30, 10, 10, 10], [40, 10, 10, 10], [50, 10, 10, 10]]
TASK = 'none'  # Options: 'none', 'mnist', or 'random'
INPUT_DIM = 10
OUTPUT_DIM = 10
BIAS_STD = 1.
EPOCHS = 1000
NUM_MEM = 1000  # Only applicable for TASK = 'random'
LR = 0.001
BATCH_SIZE = 128
OPTIMIZER = 'adam'
REPEATS = 40
HOME_DIR = './models/'
REWRITE = False  # If True, rewrites output for runs already processed

ESTIMATE_BOTH = False
KNOWN_ARCHITECTURE = True
SEED = 18
EPS = 0.001
ITERATIONS = 15
SAMPLE_NUM = 20
SAMPLE_RADIUS = int(10 * np.sqrt(INPUT_DIM))
SAMPLE_LENGTH = int(500 * np.sqrt(INPUT_DIM))
APPROX_RADIUS = 0.01
APPROX_NUM = int(1.5 * INPUT_DIM)
APPROX_THRESHOLD = 0.9
MULTIPLE_POINTS = False  # Option for tuning approx_boundary, default is False.
CHECK_RADIUS = int(100 * np.sqrt(INPUT_DIM))
CHECK_NUM = 10
CHECK_EPS = 0.001
DEDUPLICATION_EPS = 0.001
PERTURB_EPS = 0.001
PRECISION_LINE = 0.00001
PRECISION_BOUNDARY = 0.0001



# Set seeds
np.random.seed(SEED)
tf.set_random_seed(SEED)

output_file_string = 'estimate_both_%s_known_architecture_%s_seed_%d_eps_%s_iterations_%d' %(
                 str(ESTIMATE_BOTH), str(KNOWN_ARCHITECTURE), SEED, str(EPS), ITERATIONS) + \
              '_sample_radius_%d_sample_length_%d_sample_num_%d_approx_radius_%s' %(
                 SAMPLE_RADIUS, SAMPLE_LENGTH, SAMPLE_NUM, str(APPROX_RADIUS)) + \
              '_approx_num_%d_approx_threshold_%s_multiple_points_%s_check_radius_%d_check_num_%d' %(
                 APPROX_NUM, str(APPROX_THRESHOLD), str(MULTIPLE_POINTS), CHECK_RADIUS, CHECK_NUM) + \
              '_check_eps_%s_deduplication_eps_%s_perturb_eps_%s_precision_line_%s_precision_boundary_%s' %(
                 str(CHECK_EPS), str(DEDUPLICATION_EPS), str(PERTURB_EPS), str(PRECISION_LINE), str(PRECISION_BOUNDARY))
output_file = 'estimate_' + str(adler32(bytes(output_file_string, 'utf-8')))

for repeat in range(REPEATS):
    for network in NETWORKS:
        model_dir = 'task_%s_network_%s_input_dim_%d_output_dim_%d_bias_std_%s_epochs_%d_num_mem_%d_lr_%s_batch_size_%d_optimizer_%s/' % (
            TASK, list_to_str(network), INPUT_DIM, OUTPUT_DIM, str(BIAS_STD),
            EPOCHS, NUM_MEM, str(LR), BATCH_SIZE, OPTIMIZER)
        completed = os.listdir(HOME_DIR + model_dir + str(repeat))
        print("Processing network %s, run %s" %(str(network), str(repeat)))
        if (output_file not in completed) or REWRITE:
            layer1_samples = 0
            layer2_samples = 0
            with tf.Session(graph=tf.Graph()) as sess:
                tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                                           HOME_DIR + model_dir + str(repeat) + '/model')
                graph = tf.get_default_graph()

                # Prepare to find weights to layer 1.
                print("APPROXIMATING LAYER 1....")
                layer1_start = time()
                approx_weight1 = []
                approx_bias1 = []
                approx_num1 = 0
                unused = []

                # Sample line segments with midpoint at a certain radius.
                for n in range(SAMPLE_NUM):
                    midpoint = np.random.random((INPUT_DIM,)) - 0.5
                    midpoint /= np.linalg.norm(midpoint)
                    perp = np.random.random((INPUT_DIM,)) - 0.5
                    perp = perp - np.dot(midpoint, perp) * midpoint
                    perp /= np.linalg.norm(perp)
                    endpt1 = SAMPLE_RADIUS * midpoint + (SAMPLE_LENGTH / 2) * perp
                    endpt2 = SAMPLE_RADIUS * midpoint - (SAMPLE_LENGTH / 2) * perp

                    # Approximate region transition points along sample segment, and check against exact answer.
                    approx_pts, samples = approx_1d_2(sess, endpt1, endpt2, ITERATIONS, EPS, precision=PRECISION_LINE)
                    if approx_num1 == network[0] and KNOWN_ARCHITECTURE:
                        layer2_samples += samples
                    else:
                        layer1_samples += samples
                    exact_pts = calc_1d(sess, endpt1, endpt2)
                    print("Estimated num pts along line: %d, true num pts: %d" % (len(approx_pts), len(exact_pts)))

                    for point_t in approx_pts:
                        point = (endpt2 - endpt1) * point_t + endpt1

                        # Check if this point lies on a hyperplane already found.
                        repeated = False
                        for m in range(approx_num1):
                            if not repeated:
                                value = np.dot(approx_weight1[m], point) + approx_bias1[m]
                                value_threshold = CHECK_EPS * np.linalg.norm(point)
                                if value < value_threshold and value > -value_threshold:
                                    repeated = True
                        if not repeated and not (approx_num1 == network[0] and KNOWN_ARCHITECTURE):
                            # Find the boundary in the neighborhood of each point by sampling more transition points.
                            # and fitting a hyperplane.
                            weight, bias, _, pts, samples = approx_boundary(sess, point, APPROX_RADIUS, APPROX_NUM,
                                                                            APPROX_THRESHOLD, ITERATIONS, EPS,
                                                                            precision=PRECISION_BOUNDARY,
                                                                            multiple_points=MULTIPLE_POINTS)
                            layer1_samples += samples
                            if np.array(weight).shape:  # Check that a valid hyperplane was found.
                                valid = True
                                i = 0
                                extra_pts = []

                                # Samples points a certain distance away to see if the hyperplane extends that far unchanged.
                                while valid and i < CHECK_NUM:
                                    check_point = np.random.random((INPUT_DIM,)) - 0.5
                                    check_point = check_point - weight * np.dot(weight, check_point)
                                    check_point = point + CHECK_RADIUS * (check_point / np.linalg.norm(check_point))
                                    check_vec = CHECK_RADIUS * CHECK_EPS * weight
                                    straight = is_straight(sess, check_point, check_vec, EPS)
                                    layer1_samples += 3
                                    if straight:
                                        valid = False
                                        unused.append((point, weight, bias))
                                    i += 1

                                # If passes checks, add candidate hyperplane to list for layer 1.
                                if valid:
                                    # Check through proposed neurons for layer 1 for duplicate weights.
                                    duplicate = False
                                    for n in range(0, approx_num1):
                                        if np.linalg.norm(np.abs(approx_weight1[n]) - np.abs(weight)) < \
                                                DEDUPLICATION_EPS:
                                            duplicate = True
                                            break
                                    if not duplicate:
                                        approx_weight1.append(weight)
                                        approx_bias1.append(bias)
                                        approx_num1 += 1
                        elif not repeated:
                            unused.append((point, None, None))


                # Describe approximated weights in layer 1.
                approx_weight1 = np.array(approx_weight1).T
                approx_bias1 = np.array(approx_bias1)
                print("...LAYER 1 COMPLETE.")
                print("Approx num neurons in layer 1: %d, Time (seconds): %.1f" % (approx_num1, time() - layer1_start))
                print('Sample points for layer 1:', layer1_samples)

                if ESTIMATE_BOTH:
                    # Initialize for finding weights to layer 2.
                    print("APPROXIMATING LAYER 2....")
                    layer2_start = time()
                    all_sign_patterns = []
                    all_weights = []
                    all_biases = []
                    all_layer_weights = []
                    all_flip_weights = []
                    all_flip_biases = []
                    all_confirmed = []
                    num2_raw = 0
                    num2_confirmed = 0

                    # Go back to transition points that were not on boundaries for layer 1.
                    unused_i = 0
                    for (point, weight, bias) in unused:
                        unused_i += 1
                        print("Processing %d of %d ..." % (unused_i, len(unused)))

                        # Identify sign pattern of each point with respect to hyperplanes from layer 1.
                        sign_patterns = [np.sign(np.dot(point, approx_weight1) + approx_bias1)]
                        sign_pattern_hashes = [hash(str(sign_patterns[0]))]

                        # Check if this point lies on a layer 2 boundary already found.
                        repeated = False
                        for m in range(num2_raw):
                            if not repeated and all_confirmed[m]:
                                test_weight = all_weights[m][0]
                                test_bias = all_biases[m][0]
                                for q in range(approx_num1):
                                    if all_sign_patterns[m][0][q] != sign_patterns[0][q]:
                                        test_weight = test_weight + all_flip_weights[m][q, :]
                                        test_bias = test_bias + all_flip_biases[m][q]
                                value = np.dot(test_weight, point) + test_bias
                                value_threshold = CHECK_EPS * np.linalg.norm(point)  # heuristic
                                if value < value_threshold and value > -value_threshold:
                                    repeated = True
                                    print("Repeated")
                        if (not repeated) and not (num2_confirmed == network[1] and KNOWN_ARCHITECTURE):
                            # Initialize a list of points on the same (unknown) boundary.
                            if not np.array(weight).shape:
                                weight, bias, _, _, samples = approx_boundary(sess, point, APPROX_RADIUS, APPROX_NUM,
                                                                              APPROX_THRESHOLD, ITERATIONS, EPS,
                                                                              precision=PRECISION_BOUNDARY,
                                                                              multiple_points=MULTIPLE_POINTS)
                                layer2_samples += samples
                            if np.array(weight).shape:
                                weights = [weight]
                                biases = [bias]
                                points = [point]
                                explored_neurons = [[]]
                                still_to_cross = list(range(approx_num1))
                                layer_weights = np.zeros(approx_num1)
                                flip_weights = np.zeros((approx_num1, INPUT_DIM))
                                flip_biases = np.zeros(approx_num1)

                                which_to_try = [0]

                                # Travel along boundary to meet hyperplanes from layer 1.
                                i = 0
                                valid = True
                                # while valid and (i < 5 * int(CHECK_NUM * approx_num1 / INPUT_DIM) and (len(still_to_cross) > 0)):
                                while valid and (i < int(CHECK_NUM * approx_num1) and (len(still_to_cross) > 0)):
                                    i += 1
                                    done = False
                                    while not done:
                                        if len(which_to_try) == 0:
                                            # Pick a starting point from points already identified.
                                            choice = np.random.choice(len(points))
                                            start_point = points[choice]
                                            start_pattern = sign_patterns[choice]
                                            start_weight = weights[choice]
                                            start_norm = np.linalg.norm(start_weight)

                                            # Travel in a random direction along the bent hyperplane.
                                            direction = np.random.random((INPUT_DIM,)) - 0.5
                                            direction = direction - np.dot(start_weight,
                                                                           direction) * start_weight / (start_norm ** 2)
                                            # start_weight not necessarily normalized to unit norm
                                            direction = direction / np.linalg.norm(direction)
                                            done = True
                                        else:
                                            choice = np.random.choice(which_to_try)
                                            start_point = points[choice]
                                            start_pattern = sign_patterns[choice]
                                            start_weight = weights[choice]
                                            start_bias = biases[choice]
                                            start_norm = np.linalg.norm(start_weight)
                                            j = 0
                                            while (j < len(still_to_cross)) and (still_to_cross[j] in
                                                                                     explored_neurons[choice]):
                                                j += 1
                                            if j == len(still_to_cross):
                                                which_to_try.remove(choice)
                                            else:
                                                candidate = still_to_cross[j]
                                                direction = -approx_weight1[:, candidate] * start_pattern[candidate]
                                                direction = direction - np.dot(start_weight, direction) * \
                                                                        start_weight / (start_norm ** 2)
                                                # start_weight not necessarily normalized to unit norm
                                                direction = direction / np.linalg.norm(direction)
                                                done = True
                                                explored_neurons[choice].append(candidate)

                                    # Find closest points of intersection with layer 1 hyperplanes in each direction.
                                    dists = np.divide(-(np.dot(start_point, approx_weight1) + approx_bias1),
                                                      np.dot(direction, approx_weight1))
                                    inverse_dists = np.divide(1, dists)
                                    picks = []
                                    if np.max(inverse_dists) > 0:
                                        picks.append(np.argmax(inverse_dists))

                                    # For each direction, cross the layer 1 hyperplane and determine the new boundary direction.
                                    for pick in picks:
                                        if dists[pick] < CHECK_RADIUS:
                                            new_pattern = np.copy(start_pattern)
                                            new_pattern[pick] = -new_pattern[pick]
                                            if hash(str(new_pattern)) not in sign_pattern_hashes:
                                                pick_weight = np.copy(approx_weight1[:, pick])
                                                pick_bias = np.copy(approx_bias1[pick])
                                                if new_pattern[pick] < 0:
                                                    pick_weight = -pick_weight
                                                    pick_bias = -pick_bias
                                                boundary_point = start_point + dists[pick] * direction
                                                check_point = start_point + dists[pick] * direction * (1 - PERTURB_EPS)

                                                pick_dot_start = np.dot(pick_weight, start_weight)
                                                start_perp_pick = start_weight - pick_dot_start * pick_weight
                                                check_vec = dists[pick] * CHECK_EPS * start_perp_pick / np.linalg.norm(
                                                    start_perp_pick)

                                                # perp_pick = start_weight - np.dot(pick_weight,
                                                #                                   start_weight) * pick_weight
                                                # perp_pick = perp_pick / np.linalg.norm(perp_pick)
                                                # check_vec = dists[pick] * CHECK_EPS * perp_pick

                                                # pick_weight already normalized to unit norm
                                                straight = is_straight(sess, check_point, check_vec, EPS)
                                                layer2_samples += 3
                                                # If the boundary bends before layer 1 hyperplane, declare invalid.
                                                if straight:
                                                    valid = False
                                                else:

                                                    # new_weight, new_bias, new_point, samples = approx_bend(
                                                    #     sess, boundary_point, APPROX_RADIUS, APPROX_THRESHOLD,
                                                    #     ITERATIONS, EPS, (pick_weight, start_weight, perp_pick),
                                                    #     precision=PRECISION_BOUNDARY)

                                                    new_weight, new_bias, new_point, _, samples = approx_boundary(
                                                        sess, boundary_point, APPROX_RADIUS, APPROX_NUM,
                                                        APPROX_THRESHOLD, ITERATIONS, EPS,
                                                        on_pos_side_of=(pick_weight, pick_bias),
                                                        precision=PRECISION_BOUNDARY, multiple_points=MULTIPLE_POINTS)
                                                    layer2_samples += samples

                                                    if np.array(new_weight).shape:
                                                        # start_weight, new_weight, pick_weight lie in a 2D plane; consider that plane.
                                                        # Note: pick_weight and new_weight are both normalized to norm 1.
                                                        pick_2d = np.array([1, 0])
                                                        start_2d = np.array([pick_dot_start,
                                                                             np.linalg.norm(start_perp_pick)])
                                                        pick_dot_new = np.dot(pick_weight, new_weight)
                                                        new_2d = np.array([pick_dot_new,
                                                                           np.dot(new_weight, start_perp_pick) /
                                                                           np.linalg.norm(start_perp_pick)])

                                                        # Ensures that the positive sides of the new layer 2 boundary match up.
                                                        if np.dot(np.cross(start_2d, pick_2d),
                                                                  np.cross(new_2d, pick_2d)) < 0:
                                                            new_weight = -new_weight
                                                            new_bias = -new_bias

                                                        # Scales new_weight to match the scale on start_weight.
                                                        # Uses Law of Sines: if start_weight = BC, new_weight = AB, pick_weight = CA, then
                                                        # |AB| = |BC| * sin(BCA) / sin(CAB) = |BA|
                                                        cos_bca = pick_dot_start / start_norm
                                                        cos_cab = pick_dot_new
                                                        ab = start_norm * np.sqrt(1 - cos_bca ** 2) / \
                                                             np.sqrt(1 - cos_cab ** 2)
                                                        new_weight *= ab
                                                        new_bias *= ab
                                                        layer1_weight = np.dot(start_weight - new_weight, pick_weight)

                                                        # Update lists of sign patterns, weights, and biases
                                                        sign_patterns.append(new_pattern)
                                                        sign_pattern_hashes.append(hash(str(new_pattern)))
                                                        weights.append(new_weight)
                                                        biases.append(new_bias)
                                                        points.append(new_point)
                                                        if start_pattern[pick] == sign_patterns[0][pick]:
                                                            flip_weights[pick, :] = new_weight - start_weight
                                                            flip_biases[pick] = new_bias - start_bias
                                                        else:
                                                            flip_weights[pick, :] = start_weight - new_weight
                                                            flip_biases[pick] = start_bias - new_bias
                                                        which_to_try.append(len(points) - 1)
                                                        explored_neurons.append([])
                                                        layer_weights[pick] = layer1_weight
                                                        if pick in still_to_cross:
                                                            still_to_cross.remove(pick)
                                print("Neuron expected to be in layer 2:", valid)
                                print("Number of crossings identified: %d of %d" % (approx_num1 - len(still_to_cross),
                                                                                    approx_num1))

                                # If boundary is valid, add to list of layer 2 neurons.
                                if valid and len(still_to_cross) == 0:
                                    layer_weights /= np.linalg.norm(layer_weights)

                                    # Check through proposed neurons for layer 2 for duplicate weights.
                                    duplicate = False
                                    for m, prior_layer_weights in enumerate(all_layer_weights):
                                        if np.linalg.norm(np.abs(layer_weights) -
                                                                  np.abs(prior_layer_weights)) < DEDUPLICATION_EPS:
                                            if all_confirmed[m]:
                                                duplicate = True
                                                break
                                            else:
                                                all_confirmed[m] = True
                                                print("Confirmed boundary")
                                                num2_confirmed += 1
                                    if not duplicate:
                                        all_sign_patterns.append(sign_patterns)
                                        all_weights.append(weights)
                                        all_biases.append(biases)
                                        all_layer_weights.append(layer_weights)
                                        all_flip_weights.append(flip_weights)
                                        all_flip_biases.append(flip_biases)
                                        all_confirmed.append(False)
                                        num2_raw += 1

                    # Describe approximated weights in layer 2.
                    all_sign_patterns = [all_sign_patterns[m] for m in range(num2_raw) if all_confirmed[m]]
                    all_weights = [all_weights[m] for m in range(num2_raw) if all_confirmed[m]]
                    all_biases = [all_biases[m] for m in range(num2_raw) if all_confirmed[m]]
                    all_layer_weights = [all_layer_weights[m] for m in range(num2_raw) if all_confirmed[m]]
                    all_flip_weights = [all_flip_weights[m] for m in range(num2_raw) if all_confirmed[m]]
                    all_flip_biases = [all_flip_biases[m] for m in range(num2_raw) if all_confirmed[m]]
                    approx_weight2 = np.array(all_layer_weights).T
                    approx_num2 = approx_weight2.shape[1]
                    print("...LAYER 2 COMPLETE.")
                    print("Approx num neurons in layer 2: %d, Time (seconds): %.1f" % (approx_num2,
                                                                                       time() - layer2_start))
                    print('Sample points for layer 2:', layer2_samples)

                # Get true weights.
                weight1_var = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                               if w.name.endswith('kernel:0')][0]
                bias1_var = [b for b in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                             if b.name.endswith('bias:0')][0]
                weight2_var = [w for w in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                               if w.name.endswith('kernel:0')][1]
                bias2_var = [b for b in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                             if b.name.endswith('bias:0')][1]
                [true_weight1, true_bias1, true_weight2, true_bias2] = sess.run(
                    [weight1_var, bias1_var, weight2_var, bias2_var],
                    feed_dict={'X:0': np.zeros((0, INPUT_DIM)), 'Y:0': np.zeros((0,))})

            if ESTIMATE_BOTH:
                # Solve for correct signs of layer 1 hyperplanes.
                X = []
                y = []
                for n, layer_weights in enumerate(all_layer_weights):
                    for m, pattern in enumerate(all_sign_patterns[n]):
                        net_weights = np.multiply(approx_weight1, np.tile((layer_weights / 2).reshape(1, -1),
                                                                          [INPUT_DIM, 1]))
                        net_shift = -np.dot(approx_weight1, np.multiply(pattern, layer_weights / 2))
                        for p in range(1, INPUT_DIM):
                            X.append(net_weights[p, :] * all_weights[n][m][0] - net_weights[0, :] * all_weights[n][m][p])
                            y.append(net_shift[p] * all_weights[n][m][0] - net_shift[0] * all_weights[n][m][p])
                X = np.array(X)
                y = np.array(y)
                reg = RANSACRegressor().fit(X, y)
                approx_signs1 = np.sign(reg.estimator_.coef_)
                if reg.score(X, y) < APPROX_THRESHOLD:
                    print("Warning: Approximate signs for layer 1 inconsistent.")

                # Correct signs in approximated weights.
                approx_weight1 = np.multiply(approx_weight1, np.tile(approx_signs1.reshape(1, -1), [INPUT_DIM, 1]))
                approx_bias1 = np.multiply(approx_bias1, approx_signs1)

                # Solve for layer 2 biases.
                approx_bias2 = []
                for n, biases in enumerate(all_biases):
                    estimates = []
                    for m, pattern in enumerate(all_sign_patterns[n]):
                        gating = 0.5 * np.multiply(approx_signs1, pattern) + 0.5
                        scale = np.mean(np.divide(np.dot(np.multiply(approx_weight1, np.tile(
                            gating.reshape(1, -1), [INPUT_DIM, 1])), all_layer_weights[n]), all_weights[n][m]))
                        before_bias = np.dot(np.multiply(approx_bias1, gating), all_layer_weights[n])
                        estimates.append(scale * biases[m] - before_bias)
                    approx_bias2.append(np.mean(np.array(estimates)))

            # Normalize true weights to account for isomorphic networks.
            for n in range(network[0]):
                norm = np.linalg.norm(true_weight1[:, n])
                true_bias1[n] /= norm
                true_weight1[:, n] /= norm
                true_weight2[n, :] *= norm
            for n in range(network[1]):
                norm = np.linalg.norm(true_weight2[:, n])
                true_bias2[n] /= norm
                true_weight2[:, n] /= norm

            # Compare true and approximate weights in layer 1.
            true_num1 = network[0]
            unmatched_true1 = list(range(true_num1))
            unmatched_approx1 = list(range(approx_num1))
            matched_num = 0
            correct_sign_num = 0
            errs = []
            bias_errs = []
            perm1 = []
            for m in range(approx_num1):
                for n in range(true_num1):
                    for sign in [1, -1]:
                        err = np.linalg.norm(sign * approx_weight1[:, m] - true_weight1[:, n])
                        bias_err = np.linalg.norm(sign * approx_bias1[m] - true_bias1[n])
                        if err < DEDUPLICATION_EPS:
                            unmatched_true1.remove(n)
                            unmatched_approx1.remove(m)
                            matched_num += 1
                            errs.append(err)
                            bias_errs.append(bias_err)
                            if sign == 1:
                                correct_sign_num += 1
                            perm1.append(n)
            weight1_error = np.linalg.norm(np.array(errs))
            print("Layer 1 - weights L2 error:", weight1_error)
            bias1_error = np.linalg.norm(np.array(bias_errs))
            print("Layer 1 - bias L2 error:", bias1_error)
            print("Layer 1 - unmatched true neurons: %d, unmatched approx neurons: %d" % (len(unmatched_true1),
                                                                                          len(unmatched_approx1)))


            if ESTIMATE_BOTH:
                print("Layer 1 - correct signs: %d of %d" % (correct_sign_num, true_num1))
                if len(perm1) == true_num1:
                    # Permute true weights (according to approximated weights) to account for isomorphic networks.
                    true_weight1 = true_weight1[:, perm1]
                    true_bias1 = true_bias1[perm1]
                    true_weight2 = true_weight2[perm1, :]

                    # Compare true and approximate weights in layer 2.
                    true_num2 = network[1]
                    unmatched_true2 = list(range(true_num2))
                    unmatched_approx2 = list(range(approx_num2))
                    matched_num = 0
                    errs = []
                    bias_errs = []
                    perm2 = []
                    for m in range(approx_num2):
                        for n in range(true_num2):
                            for sign in [1, -1]:
                                err = np.linalg.norm(sign * approx_weight2[:, m] - true_weight2[:, n])
                                bias_err = np.linalg.norm(sign * approx_bias2[m] - true_bias2[n])
                                if err < DEDUPLICATION_EPS:
                                    unmatched_true2.remove(n)
                                    unmatched_approx2.remove(m)
                                    matched_num += 1
                                    errs.append(err)
                                    bias_errs.append(bias_err)
                                    perm2.append(n)
                    weight2_error = np.linalg.norm(np.array(errs))
                    print("Layer 2 - weights L2 error:", weight2_error)
                    bias2_error = np.linalg.norm(np.array(bias_errs))
                    print("Layer 2 - bias L2 error:", bias2_error)
                    print("Layer 2 - unmatched true neurons: %d, unmatched approx neurons: %d" %(
                        len(unmatched_true2), len(unmatched_approx2)))
                    if len(perm2) == true_num2:
                        # Permute true weights (according to approximated weights) to account for isomorphic networks.
                        true_weight2 = true_weight2[:, perm2]
                        true_bias2 = true_bias2[perm2]

            # Save output
            if ESTIMATE_BOTH:
                output = {'true_weight1': true_weight1, 'weight1_error': weight1_error,
                          'true_bias1': true_bias1, 'bias1_error': bias1_error,
                          'unmatched_true1': unmatched_true1, 'unmatched_approx1': unmatched_approx1,
                          'true_weight2': true_weight2, 'weight2_error': weight2_error,
                          'true_bias2': true_bias2, 'bias2_error': bias2_error,
                          'unmatched_true2': unmatched_true2, 'unmatched_approx2': unmatched_approx2,
                          'approx_weight1': approx_weight1, 'approx_bias1': approx_bias1,
                          'approx_weight2': approx_weight2, 'approx_bias2': approx_bias2,
                          'layer1_samples': layer1_samples, 'layer2_samples': layer2_samples}
            else:
                output = {'true_weight1': true_weight1, 'weight1_error': weight1_error,
                          'true_bias1': true_bias1, 'bias1_error': bias1_error,
                          'unmatched_true1': unmatched_true1, 'unmatched_approx1': unmatched_approx1,
                          'approx_weight1': approx_weight1, 'approx_bias1': approx_bias1,
                          'layer1_samples': layer1_samples}
            with open(HOME_DIR + model_dir + str(repeat) + '/' + output_file, 'wb') as f:
                pickle.dump(output, f)