# This module is a part of geoscan task
#
# Made by Andrey Underoak https://github.com/AndreyUnderoak

# TODO separate run to methods

import cv2
import glob as gl
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import tensorflow.compat.v1 as tf

# for python3 and tf2
tf.disable_v2_behavior()

class ModelTrainer():
    """
    Class provides training model for graying colored png images
    """
    def run():
        # model input
        model_px = 512

        # dataset path
        path = "./dataset/*.png"
        
        # create from colored dataset
        # model_pxXmodel_px colored and gray images
        # for training
        count = 0
        filenames = gl.glob(path)
        for filename in filenames:
            image = cv2.imread(filename)
            # smoothing
            color_img = cv2.GaussianBlur(cv2.resize(image, (model_px, model_px)),(5,5),0)
            # graying by cv2
            gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

            count += 1
            print("Graying image " + str(count))

            # save dataset
            cv2.imwrite("./gray_images/gray_" +str(count) +".png", gray_img)
            cv2.imwrite("./color_images/color_" +str(count) +".png", color_img)

        dataset = []

        # read all color images and append into numpy list
        for i in range(1, count):
            img = cv2.imread("color_images/color_" +str(i) +".png" )
            dataset.append(np.array(img))

        dataset_source = np.asarray(dataset)
        print(dataset_source.shape)

        dataset_tar = []

        # read all grayscale images and append into numpy list
        for i in range(1, count):
            img = cv2.imread("gray_images/gray_" +str(i) +".png", 0)    
            dataset_tar.append(np.array(img))

        dataset_target = np.asarray(dataset_tar)
        print(dataset_target.shape)

        dataset_target = dataset_target[:, :, :, np.newaxis]

        def autoencoder(inputs):
            """
            Net constructur
            """
            # encoder
            net = tf.layers.conv2d(inputs, model_px, 2, activation = tf.nn.relu)
            print(net.shape)
            net = tf.layers.max_pooling2d(net, 2, 2, padding = 'same')
            print(net.shape)


            # decoder
            net = tf.image.resize_nearest_neighbor(net, tf.constant([model_px+1, model_px+1]))
            net = tf.layers.conv2d(net, 1, 2, activation = None, name = 'outputOfAuto')
            print("NET SHAPE")
            print(net.shape)

            return net


        ae_inputs = tf.placeholder(tf.float32, (None, model_px, model_px, 3), name = 'inputToAuto')
        ae_target = tf.placeholder(tf.float32, (None, model_px, model_px, 1))

        ae_outputs = autoencoder(ae_inputs)
        lr = 0.001 # learning rate 

        loss = tf.reduce_mean(tf.square(ae_outputs - ae_target))
        train_op = tf.train.AdamOptimizer(learning_rate = lr).minimize(loss)
        init = tf.global_variables_initializer() #This will be used to run a session for training

        # define the constant variables
        batch_size_c = 5
        batch_size = batch_size_c
        epoch_num = 50
        saving_path = "pretrained/ColorToGray.ckpt"
        saver_ = tf.train.Saver(max_to_keep = 3)

        # input data to the network is stored in variables
        batch_img = dataset_source[0:batch_size]
        batch_out = dataset_target[0:batch_size]
        num_batches = count//batch_size

        # create a session object and run the global variable which was defined earlier
        sess = tf.Session()
        sess.run(init)

        # new images are sent into the network in batches of 32
        for ep in range(epoch_num):
            batch_size = 0
            for batch_n in range(num_batches): # batches loop

                # runs the computational graph in the autoencoder with the given input data and target data
                _, c = sess.run([train_op, loss], feed_dict = {ae_inputs: batch_img, ae_target: batch_out})
                print("Epoch: {} - cost = {:.5f}" .format((ep+1), c))
                batch_img = dataset_source[batch_size: batch_size+batch_size_c]
                batch_out = dataset_target[batch_size: batch_size+batch_size_c]
                batch_size += batch_size_c

            saver_.save(sess, saving_path, global_step = ep)

        sess.close()

        print("READY")

ModelTrainer.run()
