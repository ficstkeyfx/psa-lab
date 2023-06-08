#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.


import tensorflow as tf
import pdb


def lenet(input_, is_training):
    # 64 x 64 x 1
    input_ = tf.layers.average_pooling2d(input_, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 32 x 32 x 1
    conv_1 = tf.layers.conv2d(inputs=input_, filters=6, kernel_size=(5, 5), padding='valid', activation=tf.nn.relu)
    # 28 x 28 x 6
    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 14 x 14 x 6
    conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=16, kernel_size=(5, 5), padding='valid', activation=tf.nn.relu)
    # 10 x 10 x 16
    max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 5 x 5 x 16
    flat = tf.layers.flatten(max_pool_2)
    # drop_1 = tf.layers.dropout(flat, 0.3, training=is_training)
    fc_1 = tf.layers.dense(flat, units=120, activation=tf.nn.relu)
    # drop_2 = tf.layers.dropout(fc_1, 0.3, training=is_training)
    fc_2 = tf.layers.dense(fc_1, units=84, activation=tf.nn.relu)
    # drop_3 = tf.layers.dropout(fc_2, 0.3, training=is_training)
    logits = tf.layers.dense(fc_2, units=1, activation=None)
    return logits


def model_latent(input_, is_training):
    # 8 x 8 x 16
    conv_1 = tf.layers.conv2d(inputs=input_, filters=6, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # 8 x 8 x 6
    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 4 x 4 x 6
    conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # 4 x 4 x 16
    max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 2 x 2 x 16
    flat = tf.layers.flatten(max_pool_2)
    drop_1 = tf.layers.dropout(flat, 0.3, training=is_training)
    fc_1 = tf.layers.dense(drop_1, units=120, activation=tf.nn.relu)
    drop_2 = tf.layers.dropout(fc_1, 0.3, training=is_training)
    logits = tf.layers.dense(drop_2, units=1, activation=None)
    return logits


def model_latent_shallow(input_, is_training):
    # 8 x 8 x 16
    conv_1 = tf.layers.conv2d(inputs=input_, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # 8 x 8 x 6
    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 4 x 4 x 6
    conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # 4 x 4 x 16
    max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 2 x 2 x 16
    flat = tf.layers.flatten(max_pool_2)
    drop_1 = tf.layers.dropout(flat, 0.3, training=is_training)
    logits = tf.layers.dense(drop_1, units=1, activation=None)
    return logits


def model_latent_wider(input_, is_training):
    # 8 x 8 x 16
    conv_1 = tf.layers.conv2d(inputs=input_, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    conv_1 = tf.layers.conv2d(inputs=conv_1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # 8 x 8 x 6
    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 4 x 4 x 6
    conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    conv_2 = tf.layers.conv2d(inputs=conv_2, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # 4 x 4 x 16
    max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(4, 4), strides=(2, 2), padding='valid')
    # 2 x 2 x 1
    
    drop_1 = tf.layers.dropout(tf.layers.flatten(max_pool_2), 0.3, training=is_training)
    logits = tf.layers.dense(drop_1, units=1, activation=None)
    return logits


def model_latent_shallower(input_, is_training):
    # 8 x 8 x 16
    conv_1 = tf.layers.conv2d(inputs=input_, filters=8, kernel_size=(5, 5), padding='same', activation=tf.nn.relu)
    # 8 x 8 x 6
    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 4 x 4 x 6
    conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=4, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # 4 x 4 x 16
    max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 2 x 2 x 16
    flat = tf.layers.flatten(max_pool_2)
    drop_1 = tf.layers.dropout(flat, 0.3, training=is_training)
    logits = tf.layers.dense(drop_1, units=1, activation=None)
    return logits


def model_fusion_diff_and_latent(input_recon, input_latent, is_training):

    # decrease wide recon
    # 64 x 64
    conv1 = tf.layers.conv2d(inputs=input_recon, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

    # Now 64x64x32
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 32x32x32
    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 32x32x32
    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 16x16x32
    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 16x16x16
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')


    # 8 x 8 x 16
    conv_1 = tf.layers.conv2d(inputs=(encoded + input_latent), filters=6, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # 8 x 8 x 6
    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 4 x 4 x 6
    conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # 4 x 4 x 16
    max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 2 x 2 x 16
    flat = tf.layers.flatten(max_pool_2)
    drop_1 = tf.layers.dropout(flat, 0.3, training=is_training)
    fc_1 = tf.layers.dense(drop_1, units=120, activation=tf.nn.relu)
    drop_2 = tf.layers.dropout(fc_1, 0.3, training=is_training)
    logits = tf.layers.dense(drop_2, units=1, activation=None)
    return logits


def model_fusion_diff_and_latent_wider(input_recon, input_latent, is_training):

    # decrease wide recon
    # 64 x 64
    conv1 = tf.layers.conv2d(inputs=input_recon, filters=256, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

    # Now 64x64x32
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 32x32x32
    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=128, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 32x32x32
    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 16x16x32
    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 16x16x16
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')


    # 8 x 8 x 16
    conv_1 = tf.layers.conv2d(inputs=(encoded + input_latent), filters=64, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # 8 x 8 x 6
    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 4 x 4 x 6
    conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # 4 x 4 x 16
    max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 2 x 2 x 16
    flat = tf.layers.flatten(max_pool_2)
    drop_1 = tf.layers.dropout(flat, 0.3, training=is_training)
    fc_1 = tf.layers.dense(drop_1, units=120, activation=tf.nn.relu)
    drop_2 = tf.layers.dropout(fc_1, 0.3, training=is_training)
    logits = tf.layers.dense(drop_2, units=1, activation=None)
    return logits

def model_fusion_diff_and_latent_wider_latent(input_recon, input_latent, is_training):

    # decrease wide recon
    # 64 x 64
    conv1 = tf.layers.conv2d(inputs=input_recon, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)

    # Now 64x64x32
    maxpool1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 32x32x32
    conv2 = tf.layers.conv2d(inputs=maxpool1, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 32x32x32
    maxpool2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2), strides=(2, 2), padding='same')
    # Now 16x16x32
    conv3 = tf.layers.conv2d(inputs=maxpool2, filters=32, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # Now 16x16x16
    encoded = tf.layers.max_pooling2d(conv3, pool_size=(2, 2), strides=(2, 2), padding='same')


    # 8 x 8 x 16
    conv_1 = tf.layers.conv2d(inputs=(encoded + input_latent), filters=6, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # 8 x 8 x 6
    max_pool_1 = tf.layers.max_pooling2d(conv_1, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 4 x 4 x 6
    conv_2 = tf.layers.conv2d(inputs=max_pool_1, filters=16, kernel_size=(3, 3), padding='same', activation=tf.nn.relu)
    # 4 x 4 x 16
    max_pool_2 = tf.layers.max_pooling2d(conv_2, pool_size=(2, 2), strides=(2, 2), padding='valid')
    # 2 x 2 x 16
    flat = tf.layers.flatten(max_pool_2)
    drop_1 = tf.layers.dropout(flat, 0.3, training=is_training)
    fc_1 = tf.layers.dense(drop_1, units=120, activation=tf.nn.relu)
    drop_2 = tf.layers.dropout(fc_1, 0.3, training=is_training)
    logits = tf.layers.dense(drop_2, units=1, activation=None)
    return logits
