import numpy as np
import tensorflow as tf
import os
import pickle
from utils import weight_initializer, bias_initializer, list_to_str, load_mnist

NETWORKS = [[10, 10, 10], [20, 10, 10], [30, 10, 10], [40, 10, 10], [50, 10, 10]]
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

for network in NETWORKS:
    model_dir = 'task_%s_network_%s_input_dim_%d_output_dim_%d_bias_std_%s_epochs_%d_num_mem_%d_lr_%s_batch_size_%d_optimizer_%s/' %(
        TASK, list_to_str(network), INPUT_DIM, OUTPUT_DIM, str(BIAS_STD),
        EPOCHS, NUM_MEM, str(LR), BATCH_SIZE, OPTIMIZER)
    if model_dir[:-1] not in os.listdir(HOME_DIR):
        os.makedirs(HOME_DIR + model_dir)
    completed = os.listdir(HOME_DIR + model_dir)
    for repeat in range(REPEATS):
        print("Processing network %s, run %s" %(str(network), str(repeat)))
        if str(repeat) not in completed or os.listdir(HOME_DIR + model_dir + str(repeat)) == []:
            tf.reset_default_graph()
            input_placeholder = tf.placeholder(tf.float64, (None, INPUT_DIM), name='X')
            output_placeholder = tf.placeholder(tf.int64, (None,), name='Y')
            preacts = []
            relu = input_placeholder
            for width in network:
                dense_layer = tf.layers.dense(relu, width, activation=None, use_bias=True,
                                              kernel_initializer=weight_initializer,
                                              bias_initializer=bias_initializer(BIAS_STD))
                preacts.append(dense_layer)
                relu = tf.nn.relu(dense_layer)
            output_layer = tf.layers.dense(relu, OUTPUT_DIM, activation=None, use_bias=True,
                                           kernel_initializer=weight_initializer,
                                           bias_initializer=bias_initializer(BIAS_STD))
            output_layer = tf.identity(output_layer, name='output')
            with tf.Session() as sess:
                if str(repeat) not in completed:
                    os.makedirs(HOME_DIR + model_dir + str(repeat))
                if TASK == 'none':
                    sess.run(tf.global_variables_initializer())
                elif TASK == 'mnist' or TASK == 'random':
                    if TASK == 'mnist':
                        assert INPUT_DIM == 784, 'Input dimension should be 784 for MNIST.'
                        assert OUTPUT_DIM == 10, 'Output dimension should be 10 for MNIST.'
                        X_train, Y_train = load_mnist(True)
                        X_train = X_train.reshape(-1, 784)
                        X_test, Y_test = load_mnist(False)
                        X_test = X_test.reshape(-1, 784)
                    else:
                        X_train = np.random.normal(loc=0, scale=1, size=(NUM_MEM, INPUT_DIM))
                        Y_train = np.random.choice(OUTPUT_DIM, NUM_MEM, replace=True)
                        X_test = np.copy(X_train)
                        Y_test = np.copy(Y_train)
                    loss = tf.losses.sparse_softmax_cross_entropy(output_placeholder, output_layer)
                    correct = tf.equal(output_placeholder, tf.argmax(tf.nn.softmax(output_layer), axis=-1))
                    acc = 100 * tf.reduce_mean(tf.cast(correct, tf.float32))
                    if OPTIMIZER == 'adam':
                        train_op = tf.train.AdamOptimizer(LR).minimize(loss)
                    else:
                        raise NotImplementedError
                    train_losses = []
                    train_accs = []
                    test_losses = []
                    test_accs = []
                    num_training = X_train.shape[0]
                    sess.run(tf.global_variables_initializer())
                    [train_loss, train_acc] = sess.run([loss, acc], feed_dict={input_placeholder: X_train,
                                                                               output_placeholder: Y_train})
                    [test_loss, test_acc] = sess.run([loss, acc], feed_dict={input_placeholder: X_test,
                                                                             output_placeholder: Y_test})
                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                    test_losses.append(test_loss)
                    test_accs.append(test_acc)
                    i = 0
                    for i in range(1, EPOCHS + 1):
                        perm = np.random.permutation(num_training)
                        X_perm = X_train[perm, :]
                        Y_perm = Y_train[perm]
                        for j in range(int(num_training / BATCH_SIZE)):
                            X_batch = X_perm[(j * BATCH_SIZE):((j + 1) * BATCH_SIZE), :]
                            Y_batch = Y_perm[(j * BATCH_SIZE):((j + 1) * BATCH_SIZE)]
                            _ = sess.run([train_op], feed_dict={input_placeholder: X_batch, output_placeholder: Y_batch})
                        [train_loss, train_acc] = sess.run([loss, acc], feed_dict={input_placeholder: X_train,
                                                                                   output_placeholder: Y_train})
                        [test_loss, test_acc] = sess.run([loss, acc], feed_dict={input_placeholder: X_test,
                                                                                 output_placeholder: Y_test})
                        train_losses.append(train_loss)
                        train_accs.append(train_acc)
                        test_losses.append(test_loss)
                        test_accs.append(test_acc)
                    print("%d epochs, train loss %.2f, train acc %.2f, test loss %.2f, test acc %.2f" % (
                        i, train_loss, train_acc, test_loss, test_acc))
                    training = {'train_accs': train_accs, 'train_losses': train_losses,
                                'test_accs': test_accs, 'test_losses': test_losses}
                    with open(HOME_DIR + model_dir + str(repeat) + '/training', 'wb') as f:
                        pickle.dump(training, f)
                else:
                    raise NotImplementedError
                tf.saved_model.simple_save(sess, HOME_DIR + model_dir + str(repeat) + '/model',
                                           inputs={'X': input_placeholder, 'Y': output_placeholder},
                                           outputs={'output': output_layer})
