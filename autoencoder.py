from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import numpy as np
import tensorflow as tf

#Convenient class for iterating through train set randomly
class DatasetIterator:
    def __init__(self, x, y, batch_size):
        assert len(x) == len(y)
        self.x = x
        self.y = y
        self.b_sz = batch_size
        self.b_pt = 0
        self.d_sz = len(x)
        self.idx = None
        self.randomize()

    # randomize indexs in dataset
    def randomize(self):
        self.idx = np.random.permutation(self.d_sz)
        self.b_pt = 0

    # get the next batch of the dataset
    def next_batch(self):
        start = self.b_pt
        end = self.b_pt + self.b_sz
        idx = self.idx[start:end]
        x = self.x[idx]
        y = self.y[idx]
        # randomize labels
        is_labeled = np.zeros((self.b_sz, 1))
        labeled_images = np.random.permutation(np.arange(self.b_sz))[:int(0.2 * self.b_sz)]
        is_labeled[labeled_images] = 1

        self.b_pt += self.b_sz
        if self.b_pt >= self.d_sz:
            self.randomize()

        return x, y, is_labeled

def AutoEncoder(input_tensor, is_training):

    # size output of each layer
    n_inputs  = 28 * 28
    n_hidden1 = 400
    n_hidden2 = 40
    n_hidden3 = 10
    n_hidden4 = n_hidden2
    n_hidden5 = n_hidden1
    n_outputs = n_inputs # these two must be same

    x_init = tf.contrib.layers.xavier_initializer()
    z_init = tf.zeros_initializer()

    # define weights and bias for each layer
    with tf.name_scope("weights"):
        W1 = tf.get_variable(dtype=tf.float32,shape=(n_inputs,n_hidden1),initializer=x_init,name="W1")
        b1 = tf.get_variable(dtype=tf.float32,shape=(1,n_hidden1),initializer=z_init,name="b1")
        W2 = tf.get_variable(dtype=tf.float32,shape=(n_hidden1,n_hidden2),initializer=x_init,name="W2")
        b2 = tf.get_variable(dtype=tf.float32,shape=(1,n_hidden2),initializer=z_init,name="b2")
        W3 = tf.get_variable(shape=(n_hidden2,n_hidden3),initializer=x_init,name="W3")
        b3 = tf.get_variable(dtype=tf.float32,shape=(1,n_hidden3),initializer=z_init,name="b3")
        W4 = tf.get_variable(shape=(n_hidden3,n_hidden4),initializer=x_init,name="W4")
        b4 = tf.get_variable(dtype=tf.float32,shape=(1,n_hidden4),initializer=z_init,name="b4")
        W5 = tf.get_variable(shape=(n_hidden4,n_hidden5),initializer=x_init,name="W5")
        b5 = tf.get_variable(dtype=tf.float32,shape=(1,n_hidden5),initializer=z_init,name="b5")
        W6 = tf.get_variable(shape=(n_hidden5,n_outputs),initializer=x_init,name="W6")
        b6 = tf.get_variable(dtype=tf.float32,shape=(1,n_outputs),initializer=z_init,name="b6")

    with tf.name_scope("AE"):
        # encoding part
        hidden1 = tf.nn.elu(tf.matmul(input_tensor,W1)+b1)
        hidden2 = tf.nn.elu(tf.matmul(hidden1,W2)+b2)
        hidden3 = tf.nn.elu(tf.matmul(hidden2,W3)+b3)
        logits = hidden3 # this is the bottleneck
        # decoding part
        hidden4 = tf.nn.elu(tf.matmul(hidden3,W4)+b4)
        hidden5 = tf.nn.elu(tf.matmul(hidden4,W5)+b5)
        recon = tf.nn.sigmoid(tf.matmul(hidden5,W6)+b6) # sigmoid limits output within [0,1]

    return recon, logits

def run():

    # evaluate accuracy, for test and valid sets
    def evaluation(images, true_labels):
        eval_batch_size = 100
        predicted_labels = []
        # evaluate in batch sizes of 100
        for start_index in range(0, len(images), eval_batch_size):
            end_index = start_index + eval_batch_size
            batch_x = images[start_index: end_index]
            batch_predicted_labels = sess.run(prediction, feed_dict={x: batch_x, is_training: False})
            predicted_labels += list(batch_predicted_labels)
        predicted_labels = np.vstack(predicted_labels).flatten()
        true_labels = true_labels.flatten()
        accuracy = float((predicted_labels == true_labels).astype(np.int32).sum()) / len(images)
        # return predicted labels and the accuracy in comparison to the true labels
        return predicted_labels, accuracy
    
    # set iteration parameters
    EPOCHS = 10
    BATCH_SIZE = 64
    NUM_ITERS = int(55000 / BATCH_SIZE * EPOCHS)

    # get MNIST dataset
    mnist = read_data_sets('data', one_hot=False)
    x_train, y_train = (mnist.train._images, mnist.train._labels.reshape((-1, 1)))
    x_test, y_test = (mnist.test._images, mnist.test.labels.reshape((-1, 1)))
    x_valid, y_valid = (mnist.validation._images, mnist.validation._labels.reshape((-1, 1)))

    # use convenient class for iterating over MNIST dataset randomly
    train_set = DatasetIterator(x_train, y_train, BATCH_SIZE)
    test_set = DatasetIterator(x_test, y_test, BATCH_SIZE)
    valid_set = DatasetIterator(x_valid, y_valid, BATCH_SIZE)

    tf.reset_default_graph()
    x = tf.placeholder(tf.float32, (None, 784))
    y = tf.placeholder(tf.int32, (None, 1))
    is_labeled = tf.placeholder(tf.float32, (None, 1))
    is_training = tf.placeholder(tf.bool, ())
    one_hot_y = tf.one_hot(y, 10)

    # training variables
    rate = 0.001
    recon, logits = AutoEncoder(x, is_training=is_training)
    prediction = tf.argmax(logits, axis=1)
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y) * is_labeled)
    recon_loss = tf.reduce_mean((recon - x) ** 2) # loss between output of AutoEncoder and original input tensor
    loss_operation = cross_entropy + recon_loss
    optimizer = tf.train.AdamOptimizer(learning_rate=rate)
    grads_and_vars = optimizer.compute_gradients(loss_operation, tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    training_operation = optimizer.apply_gradients(grads_and_vars)

    # train
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    print("Training...")
    for i in range(NUM_ITERS):
        # get next batch
        batch_x, batch_y, batch_is_labeled = train_set.next_batch()
        # evaluate next batch
        _ = sess.run(training_operation, feed_dict={x: batch_x, y: batch_y, is_labeled: batch_is_labeled, is_training: True})
        if (i + 1) % 1000 == 0 or i == NUM_ITERS - 1:
            # validate
            _, validation_accuracy = evaluation(valid_set.x, valid_set.y)
            print("Iter {}: Validation Accuracy = {:.3f}".format(i, validation_accuracy))
    # test
    print('Evaluating on test set')
    _, test_accuracy = evaluation(test_set.x, test_set.y)
    print("Test Accuracy = {:.3f}".format(test_accuracy))

    sess.close()
    return test_accuracy


if __name__ == '__main__':
    run()
