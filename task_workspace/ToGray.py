import cv2
import glob as gl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

def autoencoder(inputs):

    # Encoder
    net = tf.layers.conv2d(inputs, 128, 2, activation = tf.nn.relu)
    print(net.shape)
    net = tf.layers.max_pooling2d(net, 2, 2, padding = 'same')
    print(net.shape)


    # Decoder
    net = tf.image.resize_nearest_neighbor(net, tf.constant([129, 129]))
    net = tf.layers.conv2d(net, 1, 2, activation = None, name = 'outputOfAuto')
    print("NET SHAPE")
    print(net.shape)

    return net

class ToGray:
    def __init__(self, path_to_model, filenames):

        ae_inputs = tf.placeholder(tf.float32, (None, 128, 128, 3), name = 'inputToAuto')
        ae_target = tf.placeholder(tf.float32, (None, 128, 128, 1))
        ae_outputs = autoencoder(ae_inputs)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        saver.restore(sess, path_to_model)

        test_data = []
        for file in filenames:
            test_data.append(np.array(cv2.resize(cv2.imread(file), (128, 128)))[:,:,:3])

        test_dataset = np.asarray(test_data)    
        batch_imgs = test_dataset
        gray_imgs = sess.run(ae_outputs, feed_dict = {ae_inputs: batch_imgs})

        for i in range(gray_imgs.shape[0]):
            cv2.imwrite('output_images/o' +str(i) +'.png', cv2.resize(gray_imgs[i],(2048, 2048)))