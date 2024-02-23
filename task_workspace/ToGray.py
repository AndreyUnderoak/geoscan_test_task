import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

class Model:
    def __init__(self, px):
        self.px = px

    def autoencoder(self, inputs):

        # Encoder
        net = tf.layers.conv2d(inputs, self.px, 2, activation = tf.nn.relu)
        print(net.shape)
        net = tf.layers.max_pooling2d(net, 2, 2, padding = 'same')
        print(net.shape)


        # Decoder
        net = tf.image.resize_nearest_neighbor(net, tf.constant([self.px+1, self.px+1]))
        net = tf.layers.conv2d(net, 1, 2, activation = None, name = 'outputOfAuto')
        print("NET SHAPE")
        print(net.shape)

        return net
    
    def get_ae_inputs_form(self):
        return tf.placeholder(tf.float32, (None, self.px, self.px, 3), name = 'inputToAuto')

    

class ToGray:
    def __init__(self, path_to_model, px):
        self.px = px
        self.model = Model(px)
        self.ae_inputs = self.model.get_ae_inputs_form()
        self.ae_outputs = self.model.autoencoder(self.ae_inputs)

        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(self.init)
        self.saver.restore(self.sess, path_to_model)

    def color_to_gray_array(self, images_np):
        batch_imgs = images_np
        gray_imgs = self.sess.run(self.ae_outputs, feed_dict = {self.ae_inputs: batch_imgs})
        return gray_imgs

    


