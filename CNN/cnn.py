import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.training import moving_averages
from tensorflow.python.ops import control_flow_ops
import gc
import time
import os
import cv2

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


max_steps = 2500
batch_size = 16

# 125 steps 遍历一遍数据

learning_rate = 0.001
learning_decay_rate = 0.999

train_files = os.listdir("data_for_cnn/train")
test_files = os.listdir("data_for_cnn/test")

num_train = len(train_files)
num_test = len(test_files)

training_step = tf.Variable(0, trainable=False)

x = tf.placeholder(tf.float32, [None, 150,150, 3])
y_ = tf.placeholder(tf.int32, [None])

is_training = tf.placeholder(tf.bool)

x = tf.image.random_flip_left_right(x)

def variable_with_weight_loss(shape, stddev, wl):
    var = tf.Variable(tf.truncated_normal(shape, stddev= stddev))
    if wl is not None:
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wl, name = "weight_loss")
        tf.add_to_collection("losses",weight_loss)
    return var

def BN(x, is_training , BN_decay = 0.9, BN_EPSILON = 1e-05):
    x_shape = x.get_shape()
    params_shapes = x_shape[-1:]

    axis = list(range(len(x_shape)-1))

    beta = tf.Variable(tf.zeros(shape=params_shapes))
    gamma = tf.Variable(tf.ones(shape=params_shapes))

    moving_mean = tf.Variable(tf.zeros(shape=params_shapes), name="mean",trainable=False)
    moving_variance = tf.Variable(tf.ones(shape=params_shapes), name = "variance",trainable=False)

    mean, variance = tf.nn.moments(x, axis)

    update_moving_mean = moving_averages.assign_moving_average(moving_mean, mean, BN_decay)
    update_moving_variance = moving_averages.assign_moving_average(moving_variance, variance, BN_decay)

    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

    mean, variance = control_flow_ops.cond(is_training, lambda:(mean, variance), lambda :(moving_mean, moving_variance))

    return tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)


# 卷积神经网络 卷积部分
kernel1 = variable_with_weight_loss(shape=[3,3,3,32], stddev=0.05, wl = 0.0)
conv1 = tf.nn.conv2d(x, kernel1, [1,1,1,1], padding="SAME")
bias1 = tf.Variable(tf.constant(0.0, shape=[32]))
bn1 = BN(tf.nn.bias_add(conv1, bias1), is_training = is_training)
relu1 = tf.nn.relu(bn1)
# relu1 = tf.nn.relu(tf.nn.bias_add(conv1, bias1))
pool1 = tf.nn.max_pool(relu1, ksize=[1,2,2,1], strides=[1,1,1,1],padding="SAME")

kernel2 = variable_with_weight_loss(shape=[3,3,32,32], stddev=0.05, wl = 0.0)
conv2 = tf.nn.conv2d(pool1, kernel2, [1,1,1,1], padding="SAME")
bias2 = tf.Variable(tf.constant(0.1, shape=[32]))
bn2 = BN(tf.nn.bias_add(conv2, bias2), is_training= is_training)
relu2 = tf.nn.relu(bn2)
# relu2 = tf.nn.relu(tf.nn.bias_add(conv2, bias2))
pool2 = tf.nn.max_pool(relu2, ksize=[1,2,2,1], strides=[1,1,1,1],padding="SAME")

kernel3 = variable_with_weight_loss(shape=[3,3,32,64], stddev=0.05, wl = 0.0)
conv3 = tf.nn.conv2d(pool2, kernel3, [1,1,1,1], padding="SAME")
bias3 = tf.Variable(tf.constant(0.1, shape=[64]))
bn3 = BN(tf.nn.bias_add(conv3, bias3), is_training = is_training)
relu3 = tf.nn.relu(bn3)
# relu3 = tf.nn.relu(tf.nn.bias_add(conv3, bias3))
pool3 = tf.nn.max_pool(relu3, ksize=[1,3,3,1], strides=[1,2,2,1],padding="SAME")

kernel4 = variable_with_weight_loss(shape=[3,3,64,64], stddev=0.05, wl = 0.0)
conv4 = tf.nn.conv2d(pool3, kernel4, [1,1,1,1], padding="SAME")
bias4 = tf.Variable(tf.constant(0.1, shape=[64]))
bn4 = BN(tf.nn.bias_add(conv4, bias4), is_training=is_training)
relu4 = tf.nn.relu(bn4)
# relu4 = tf.nn.relu(tf.nn.bias_add(conv4, bias4))
pool4 = tf.nn.max_pool(relu4, ksize=[1,3,3,1], strides=[1,2,2,1],padding="SAME")

# bsize = x.get_shape()[0].value
# print(x)

psize = pool4.get_shape()
reshape = tf.reshape(pool4, [-1, psize[1]*psize[2]*psize[3]])
dim = reshape.get_shape()[1].value

# 全相连神经网络部分
weight1 = variable_with_weight_loss([dim, 128], stddev=0.04, wl = 0.004)
fc_bias1 = tf.Variable(tf.constant(0.1, shape = [128]))
# fc_1 = tf.nn.relu(tf.matmul(reshape, weight1) + fc_bias1)

weight2 = variable_with_weight_loss([128, 64], stddev=0.04, wl = 0.004)
fc_bias2 = tf.Variable(tf.constant(0.1, shape = [64]))
# fc_2 = tf.nn.relu(tf.matmul(fc_1, weight2) + fc_bias2)

weight3 = variable_with_weight_loss([64,2], stddev=1/64, wl = 0.0)
fc_bias3 = tf.Variable(tf.constant(0.1, shape=[2]))
# result = tf.add(tf.matmul(fc_2, weight3), fc_bias3)

dro = tf.placeholder(tf.float32)

def hidden_layer(input_tensor, w1,b1, w2,b2, w3,b3, dro,layer_name):
    fc1 = tf.nn.relu(tf.matmul(input_tensor,w1) + b1)
    fc_1_droped = tf.nn.dropout(fc1, dro)

    fc2 = tf.nn.relu(tf.matmul(fc_1_droped, w2) + b2)
    fc_2_droped = tf.nn.dropout(fc2, dro)

    return tf.add(tf.matmul(fc_2_droped, w3), b3)

#全相连网络部分
result = hidden_layer(reshape, weight1,fc_bias1,weight2,fc_bias2,
                      weight3,fc_bias3,dro = dro,layer_name="y")


# 滑动平均值部分
# averages_class = tf.train.ExponentialMovingAverage(0.99, training_step)
# averages_op = averages_class.apply(tf.trainable_variables())
# averages_y = hidden_layer(reshape, averages_class.average(weight1),averages_class.average(fc_bias1),
#                           averages_class.average(weight2),averages_class.average(fc_bias2),
#                           averages_class.average(weight3),averages_class.average(fc_bias3),
#                           layer_name="average_y")

# 损失值
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=result,
                                                               labels=tf.cast(y_, tf.int32))
learn_rate = tf.train.exponential_decay(learning_rate, training_step, num_train/batch_size*3,
                                        learning_decay_rate)

weight_with_l2_loss = tf.add_n(tf.get_collection("losses"))
loss = tf.reduce_mean(cross_entropy) + weight_with_l2_loss
# loss = tf.reduce_mean(cross_entropy)

# train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, global_step=training_step)
# train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)

update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_op):
    train_step = tf.train.GradientDescentOptimizer(learn_rate).minimize(loss, global_step=training_step)


def train():
    train_accuracys = []
    test_accuracys = []
    train_losses = []
    test_losses = []

    saver = tf.train.Saver()
    with tf.Session() as sess:
    # with InteractiveSession(config=config) as sess:
        tf.global_variables_initializer().run()
        # tf.train.start_queue_runners()

        circle = int(num_train/batch_size)

        for step in range(max_steps):
            start_time = time.time()
            i = int(step%circle)

            files = train_files[i*batch_size: i*batch_size + batch_size]
            x_batch = []
            y_batch = []
            for file in files:
                img = cv2.imread("data_for_cnn/train/"+file)
                label = int(file[0])
                x_batch.append(img)
                y_batch.append(label)
            x_batch = np.array(x_batch, dtype=np.float32)
            y_batch = np.array(y_batch, dtype=np.uint8)

            print(step,i*batch_size, i*batch_size + batch_size)
            # sess.run(train_op, feed_dict={x:x_batch, y_:y_batch})
            sess.run(train_step, feed_dict={x:x_batch, y_:y_batch, dro:1, is_training:True})
            duration = time.time() - start_time


            if step%125 == 0:

                gc.collect()

                examples_per_sec = batch_size/duration
                sec_per_batch = float(duration)

                true_count = 0
                total_test_num = 32*batch_size

                x_train_batches , y_train_batches = random_batches(32, "train")

                for x_batch,y_batch in zip(x_train_batches,y_train_batches):
                    predicition,train_loss = sess.run([result,loss], feed_dict={x: x_batch, y_: y_batch, dro:1, is_training:False})
                    predicition = np.argmax(predicition, axis=1)
                    predicition = (predicition == y_batch.astype("int32"))
                    # print(predicition)
                    true_count += np.sum(predicition.astype("int32"))

                print(true_count,total_test_num)
                train_acc = true_count / total_test_num
                train_accuracys.append(train_acc)
                train_losses.append(train_loss)
                true_count = 0
                total_test_num = 16*batch_size

                x_test_batches, y_test_batches = random_batches(16, "test")

                for x_batch,y_batch in zip(x_test_batches,y_test_batches):
                    predicition, test_loss = sess.run([result, loss], feed_dict={x: x_batch, y_: y_batch, dro:1, is_training:False})
                    predicition = np.argmax(predicition, axis=1)
                    predicition = (predicition == y_batch.astype("int32"))
                    # print(predicition)
                    true_count += np.sum(predicition.astype("int32"))

                test_acc = true_count/total_test_num
                test_accuracys.append(test_acc)
                test_losses.append(test_loss)

                print(true_count,total_test_num)
                print("step %d,train accuracy = %.2f, test accuracy = %.2f (%.1f examples/sec; %.3f sec/batch)" %
                      (step, train_acc*100, test_acc*100 , examples_per_sec, sec_per_batch))
        # saver.save(sess, "model/cnn/cnn.ckpt")

    return train_accuracys, test_accuracys, train_losses, test_losses

def random_batches(size,type):
    if type == "train":
        files = train_files
        indices = np.random.choice(range(num_train), size * batch_size)
    else:
        files = test_files
        indices = np.random.choice(range(num_test), size * batch_size)
    print(indices)
    # print(type, indices)
    x_batches = []
    y_batches = []
    for i in range(size):
        x_batch = []
        y_batch = []
        for j in indices[i * batch_size:i * batch_size + batch_size]:
            file = files[j]
            img = cv2.imread("data_for_cnn/"+type+"/" + file)
            label = int(file[0])
            x_batch.append(img)
            y_batch.append(label)

        x_batch = np.array(x_batch, dtype=np.float32)
        y_batch = np.array(y_batch, dtype=np.uint8)
        # print(x_batch.shape, y_batch.shape)

        x_batches.append(x_batch)
        y_batches.append(y_batch)

    return x_batches,y_batches




if __name__ == '__main__':
    train_accuracys, test_accuracys, train_losses, test_losses = train()
    #
    x = range(int(max_steps/125))
    plt.plot(x,train_accuracys, label = "train_acc", color = "r")
    plt.plot(x,test_accuracys, label = "test_acc")
    plt.xlabel("step")
    plt.ylabel("acc")
    plt.title("accuracys")
    plt.legend()
    plt.show()
    #
    plt.plot(x, train_losses, label="train_loss", color="r")
    plt.plot(x, test_losses, label="test_loss")
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("loss")
    plt.legend()
    plt.show()

    # sess1 = tf.Session()
    #
    # sess1.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # saver.restore(sess1,"model/cnn/cnn.ckpt")
    #
    # writer = tf.summary.FileWriter("log/cnn", tf.get_default_graph())
    # writer.close()

    # datapath = "data_for_cnn/test/"
    # count = 0
    # acc = 0
    #
    # files = os.listdir(datapath)
    # for file in files:
    #     file_path = datapath + file
    #     img = cv2.imread(file_path)
    #     res = sess1.run(result, feed_dict={x: [img], is_training: False, dro: 1})
    #     label = int(file[0])
    #     print(file_path, np.argmax(res),label)
    #     count += 1
    #     if (np.argmax(res) == label):
    #         acc += 1
    #
    # print(count, acc, acc/count)
    # sess1.close()