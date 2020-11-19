import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.layers import xavier_initializer_conv2d, xavier_initializer
import gc
import math
import time
import os
import cv2

batch_size = 8
max_step = 10000

learning_rate = 0.001
learning_decay = 0.999

# train_files = os.listdir("data_for_cnn/train")
# test_files = os.listdir("data_for_cnn/test")

train_files = os.listdir("/home/zzh/PycharmProjects/coal_mg_recongize/CNN/data_for_cnn/train")
test_files = os.listdir("/home/zzh/PycharmProjects/coal_mg_recongize/CNN/data_for_cnn/test")

num_train = len(train_files)
num_test = len(test_files)

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class VGG(object):
    def __init__(self):
        batch_size = 8
        max_step = 10000

        learning_rate = 0.001
        learning_decay = 0.999

        # train_files = os.listdir("/home/zzh/PycharmProjects/coal_mg_recongize/CNN/data_for_cnn/train")
        # test_files = os.listdir("/home/zzh/PycharmProjects/coal_mg_recongize/CNN/data_for_cnn/test")
        #
        # num_train = len(train_files)
        # num_test = len(test_files)

        self.training_step = tf.Variable(0, trainable=False)

        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

        def BN(x, is_training, BN_decay=0.9, BN_EPSILON=1e-05):
            x_shape = x.get_shape()
            params_shapes = x_shape[-1:]

            axis = list(range(len(x_shape) - 1))

            beta = tf.Variable(tf.zeros(shape=params_shapes))
            gamma = tf.Variable(tf.ones(shape=params_shapes))

            moving_mean = tf.Variable(tf.zeros(shape=params_shapes), name="mean", trainable=False)
            moving_variance = tf.Variable(tf.ones(shape=params_shapes), name="variance", trainable=False)

            mean, variance = tf.nn.moments(x, axis)

            update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_decay)
            update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_decay)

            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

            mean, variance = control_flow_ops.cond(is_training, lambda: (mean, variance),
                                                   lambda: (moving_mean, moving_variance))

            return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)

        def conv_op(input, name, kernel_h, kernel_w, num_out, step_h, step_w, para, is_training):
            num_in = input.get_shape()[-1].value
            with tf.name_scope(name) as scope:
                kernel = tf.get_variable(scope + "w", shape=[kernel_h, kernel_w, num_in, num_out], dtype=tf.float32,
                                         initializer=xavier_initializer_conv2d())
                conv = tf.nn.conv2d(input, kernel, (1, step_h, step_w, 1), padding="SAME")
                biases = tf.Variable(tf.constant(0.0, shape=[num_out], dtype=tf.float32), trainable=True, name="b")
                bn = BN(tf.nn.bias_add(conv, biases), is_training=is_training)
                activation = tf.nn.relu(bn, name=scope)
                para += [kernel, biases]
                return activation
            # train_accuracys, test_accuracys, train_losses, test_losses = train()

        def variable_with_weight_loss(name, shape, wl=None):
            var = tf.get_variable(name=name, shape=shape, dtype=tf.float32, initializer=xavier_initializer())
            if wl is not None:
                weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name="weight_loss")
                tf.add_to_collection("losses", weight_loss)
            return var

        def fc_op(input, name, num_out, para, wl=None):
            num_in = input.get_shape()[-1].value
            with tf.name_scope(name) as scope:
                # weights = tf.get_variable(scope + "w", shape=[num_in, num_out], dtype=tf.float32,
                #                           initializer=xavier_initializer())
                weights = variable_with_weight_loss(name=scope + "w", shape=[num_in, num_out], wl=wl)
                biases = tf.Variable(tf.constant(0.1, shape=[num_out], dtype=tf.float32), name="b")
                # activation = tf.nn.relu(tf.nn.bias_add(tf.matmul(input, weights),  biases))
                # activation = tf.nn.relu_layer(input, weights, biases)
                para += [weights, biases]
                return tf.nn.bias_add(tf.matmul(input, weights), biases)

        def inference_op(input, keep_prob, is_training):
            parameters = []

            conv1_1 = conv_op(input, name="conv1_1", kernel_h=3, kernel_w=3, num_out=64, step_h=1, step_w=1,
                              para=parameters,
                              is_training=is_training)
            conv1_2 = conv_op(conv1_1, name="conv1_2", kernel_h=3, kernel_w=3, num_out=64, step_h=1, step_w=1,
                              para=parameters,
                              is_training=is_training)
            pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool1")

            conv2_1 = conv_op(pool1, name="conv2_1", kernel_h=3, kernel_w=3, num_out=128, step_h=1, step_w=1,
                              para=parameters,
                              is_training=is_training)
            conv2_2 = conv_op(conv2_1, name="conv2_2", kernel_h=3, kernel_w=3, num_out=128, step_h=1, step_w=1,
                              para=parameters,
                              is_training=is_training)
            pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool2")

            conv3_1 = conv_op(pool2, name="conv3_1", kernel_h=3, kernel_w=3, num_out=256, step_h=1, step_w=1,
                              para=parameters,
                              is_training=is_training)
            conv3_2 = conv_op(conv3_1, name="conv3_2", kernel_h=3, kernel_w=3, num_out=256, step_h=1, step_w=1,
                              para=parameters,
                              is_training=is_training)
            conv3_3 = conv_op(conv3_2, name="conv3_3", kernel_h=3, kernel_w=3, num_out=256, step_h=1, step_w=1,
                              para=parameters,
                              is_training=is_training)
            pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool3")

            conv4_1 = conv_op(pool3, name="conv4_1", kernel_h=3, kernel_w=3, num_out=512, step_h=1, step_w=1,
                              para=parameters,
                              is_training=is_training)
            conv4_2 = conv_op(conv4_1, name="conv4_2", kernel_h=3, kernel_w=3, num_out=512, step_h=1, step_w=1,
                              para=parameters,
                              is_training=is_training)
            conv4_3 = conv_op(conv4_2, name="conv4_3", kernel_h=3, kernel_w=3, num_out=512, step_h=1, step_w=1,
                              para=parameters,
                              is_training=is_training)
            pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool4")

            conv5_1 = conv_op(pool4, name="conv5_1", kernel_h=3, kernel_w=3, num_out=512, step_h=1, step_w=1,
                              para=parameters,
                              is_training=is_training)
            conv5_2 = conv_op(conv5_1, name="conv5_2", kernel_h=3, kernel_w=3, num_out=512, step_h=1, step_w=1,
                              para=parameters,
                              is_training=is_training)
            conv5_3 = conv_op(conv5_2, name="conv5_3", kernel_h=3, kernel_w=3, num_out=512, step_h=1, step_w=1,
                              para=parameters,
                              is_training=is_training)
            pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name="pool5")

            pool_shape = pool5.get_shape().as_list()
            flattened_shape = pool_shape[1] * pool_shape[2] * pool_shape[3]
            reshaped = tf.reshape(pool5, [-1, flattened_shape], name="reshaped")

            fc_6 = fc_op(reshaped, name="fc6", num_out=2048, para=parameters, wl=0.004)
            fc_6_relu = tf.nn.relu(fc_6)
            fc_6_droped = tf.nn.dropout(fc_6_relu, keep_prob=keep_prob, name="fc6_drop")

            fc_7 = fc_op(fc_6_droped, name="fc7", num_out=2048, para=parameters, wl=0.004)
            fc_7_relu = tf.nn.relu(fc_7)
            fc_7_droped = tf.nn.dropout(fc_7_relu, keep_prob=keep_prob, name="fc7_drop")

            fc_8 = fc_op(fc_7_droped, name="fc_8", num_out=2, para=parameters)
            return fc_8

        self.x = tf.placeholder(shape=[None, 150, 150, 3], dtype=tf.float32)
        self.y_ = tf.placeholder(shape=[None], dtype=tf.int32)
        self.x = tf.image.random_flip_left_right(self.x)
        self.is_training = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32)
        self.result = inference_op(self.x, keep_prob=self.keep_prob, is_training=self.is_training)

        cross_entrpy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.result, labels=tf.cast(self.y_, tf.int32))
        learn_rate = tf.train.exponential_decay(learning_rate, self.training_step, num_train / batch_size, learning_decay)

        weight_with_l2_loss = tf.add_n(tf.get_collection("losses"))
        self.loss = tf.reduce_mean(cross_entrpy) + weight_with_l2_loss

        # loss = tf.reduce_mean(cross_entrpy)
        update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_op):
            self.train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(self.loss, global_step=self.training_step)

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def train(self):

        train_accuracys = []
        test_accuracys = []
        train_losses = []
        test_losses = []

        saver = tf.train.Saver()
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            # tf.train.start_queue_runners()

            vars = tf.global_variables()
            for var in vars:
                print(var)

            circle = int(num_train / batch_size)

            for step in range(max_step):
                start_time = time.time()
                i = int(step % circle)

                files = train_files[i * batch_size: i * batch_size + batch_size]
                x_batch = []
                y_batch = []
                for file in files:
                    img = cv2.imread("/home/zzh/PycharmProjects/coal_mg_recongize/CNN/data_for_cnn/train/" + file)
                    label = int(file[0])
                    x_batch.append(img)
                    y_batch.append(label)
                x_batch = np.array(x_batch, dtype=np.float32)
                y_batch = np.array(y_batch, dtype=np.uint8)

                print(step, i * batch_size, i * batch_size + batch_size)
                # sess.run(train_op, feed_dict={x:x_batch, y_:y_batch})
                sess.run(self.train_step, feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: 0.5, self.is_training: True})
                duration = time.time() - start_time

                # if step%125 == 0:
                if step % 250 == 0:
                    gc.collect()

                    examples_per_sec = batch_size / duration
                    sec_per_batch = float(duration)

                    true_count = 0
                    total_test_num = 32 * batch_size

                    x_train_batches, y_train_batches = self.random_batches(32, "train")

                    for x_batch, y_batch in zip(x_train_batches, y_train_batches):
                        predicition, train_loss = sess.run([self.result, self.loss],
                                                           feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: 1,
                                                                      self.is_training: False})
                        predicition = np.argmax(predicition, axis=1)
                        predicition = (predicition == y_batch.astype("int32"))
                        # print(predicition)
                        true_count += np.sum(predicition.astype("int32"))

                    print(true_count, total_test_num)
                    train_acc = true_count / total_test_num
                    train_accuracys.append(train_acc)
                    train_losses.append(train_loss)
                    true_count = 0
                    total_test_num = 16 * batch_size

                    x_test_batches, y_test_batches = self.random_batches(16, "test")

                    for x_batch, y_batch in zip(x_test_batches, y_test_batches):
                        predicition, test_loss = sess.run([self.result, self.loss],
                                                          feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: 1,
                                                                     self.is_training: False})
                        predicition = np.argmax(predicition, axis=1)
                        predicition = (predicition == y_batch.astype("int32"))
                        # print(predicition)
                        true_count += np.sum(predicition.astype("int32"))

                    test_acc = true_count / total_test_num
                    test_accuracys.append(test_acc)
                    test_losses.append(test_loss)

                    print(true_count, total_test_num)
                    print("step %d,train accuracy = %.2f, test accuracy = %.2f (%.1f examples/sec; %.3f sec/batch)" %
                          (step, train_acc * 100, test_acc * 100, examples_per_sec, sec_per_batch))
            saver.save(sess, "/home/zzh/PycharmProjects/coal_mg_recongize/CNN/model/vgg16/vgg16.ckpt")

        return train_accuracys, test_accuracys, train_losses, test_losses

    def random_batches(self,size, type):
        if type == "train":
            files = train_files
            indices = np.random.choice(range(num_train), size * batch_size)
        else:
            files = test_files
            indices = np.random.choice(range(num_test), size * batch_size)
        # print(indices)
        # print(type, indices)
        x_batches = []
        y_batches = []
        for i in range(size):
            x_batch = []
            y_batch = []
            for j in indices[i * batch_size:i * batch_size + batch_size]:
                file = files[j]
                img = cv2.imread("/home/zzh/PycharmProjects/coal_mg_recongize/CNN/data_for_cnn/" + type + "/" + file)
                label = int(file[0])
                x_batch.append(img)
                y_batch.append(label)

            x_batch = np.array(x_batch, dtype=np.float32)
            y_batch = np.array(y_batch, dtype=np.uint8)
            # print(x_batch.shape, y_batch.shape)

            x_batches.append(x_batch)
            y_batches.append(y_batch)

        return x_batches, y_batches

    def load(self,path="/home/zzh/PycharmProjects/coal_mg_recongize/CNN/model/vgg16/vgg16.ckpt"):
        saver = tf.train.Saver()
        saver.restore(self.sess,path)

    def predict(self,inputs):
        res = self.sess.run(self.result,feed_dict={self.x: inputs, self.is_training: False, self.keep_prob: 1})
        return np.argmax(res,axis=1)
        pass

    def close(self):
        self.sess.close()
    def open(self):
        try:
            self.sess.close()
        except Exception as e:
            print("sess已关闭")
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())
    def __del__(self):
        try:
            self.sess.close()
        except Exception as e:
            pass

# if __name__ == '__main__':
    # train_accuracys, test_accuracys, train_losses, test_losses = train()
    #
    #
    # x = range(int(max_step/250))
    # plt.plot(x,train_accuracys, label = "train_acc", color = "r")
    # plt.plot(x,test_accuracys, label = "test_acc")
    # plt.xlabel("step")
    # plt.ylabel("acc")
    # plt.title("accuracys")
    # plt.legend()
    # plt.show()
    #
    # plt.plot(x, train_losses, label="train_loss", color="r")
    # plt.plot(x, test_losses, label="test_loss")
    # plt.xlabel("step")
    # plt.ylabel("loss")
    # plt.title("loss")
    # plt.legend()
    # plt.show()

    # sess1 = tf.Session()
    #
    # sess1.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # saver.restore(sess1, "model/vgg16/vgg16.ckpt")
    #
    # datapath = "data_for_cnn/test/"
    # count = 0
    # acc = 0
    #
    # files = os.listdir(datapath)
    # for file in files:
    #     file_path = datapath + file
    #     img = cv2.imread(file_path)
    #     res = sess1.run(result, feed_dict={x: [img], is_training: False, keep_prob: 1})
    #     label = int(file[0])
    #     print(file_path, np.argmax(res), label)
    #     count += 1
    #     if (np.argmax(res) == label):
    #         acc += 1
    #
    # print(count, acc, acc / count)
    # sess1.close()

    # vgg = VGG()
    # vgg.train()























