#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.


import tensorflow as tf
import numpy as np
import pdb


def channel_attention_module(input_, ratio):
    # input_ size B x H x W x C
    with tf.name_scope('cam') as scope:
        C = int(input_.shape[3])
        f_avg = tf.contrib.layers.avg_pool2d(input_, kernel_size=(np.int(input_.shape[1]), np.int(input_.shape[2])),
                                             stride=1, padding='VALID')

        f_max = tf.contrib.layers.max_pool2d(input_, kernel_size=(np.int(input_.shape[1]), np.int(input_.shape[2])),
                                             stride=1, padding='VALID')

        new_c = C // ratio
        weights_0 = tf.get_variable("weights_0", (1, 1, C, new_c))
        biases_0 = tf.get_variable("biases_0", new_c)
        weights_1 = tf.get_variable("weights_1", (1, 1, new_c, C))
        biases_1 = tf.get_variable("biases_1", C)

        f_a_1 = tf.nn.relu(tf.nn.conv2d(f_avg, filter=weights_0, strides=[1, 1, 1, 1], padding="VALID") + biases_0)
        f_a_2 = tf.nn.conv2d(f_a_1, filter=weights_1, strides=[1, 1, 1, 1], padding="VALID") + biases_1

        f_m_1 = tf.nn.relu(tf.nn.conv2d(f_max, filter=weights_0, strides=[1, 1, 1, 1], padding="VALID") + biases_0)
        f_m_2 = tf.nn.conv2d(f_m_1, filter=weights_1, strides=[1, 1, 1, 1], padding="VALID") + biases_1

        m_c = tf.nn.sigmoid(f_a_2 + f_m_2)
        x_tilde = tf.multiply(input_, m_c)

        return x_tilde


def spatial_attention_module(input_):
    # input_ size B x H x W x C
    with tf.name_scope('sam') as scope:
        f_avg = tf.reduce_mean(input_, axis=-1)
        f_max = tf.math.reduce_max(input_, axis=-1)
        f_pool = tf.stack((f_avg, f_max), axis=-1)
        conv_layer = tf.layers.conv2d(inputs=f_pool, filters=1, kernel_size=(3, 3), padding='same',
                                      activation=tf.nn.sigmoid)
        # B x H x W 
        x_tilde = tf.multiply(input_, conv_layer)

        return x_tilde


def create_mask(input_):
    # input_ size B x H x W x C
    with tf.name_scope('mask') as scope:
        f_avg = tf.reduce_mean(input_, axis=-1)
        f_max = tf.math.reduce_max(input_, axis=-1)
        f_pool = tf.stack((f_avg, f_max), axis=-1)
        mask = tf.layers.conv2d(inputs=f_pool, filters=1, kernel_size=(3, 3), padding='same',  activation=None)
        return mask

def cbam_module(inputs_, ratio=4):
    cam = channel_attention_module(inputs_, ratio=ratio)
    sam = spatial_attention_module(cam)
    return sam
