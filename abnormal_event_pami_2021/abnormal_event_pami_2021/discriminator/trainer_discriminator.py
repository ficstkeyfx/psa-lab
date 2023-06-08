#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.


import tensorflow as tf
import numpy as np
import datetime
import os
import sys
import re
import pdb
from sklearn.metrics import confusion_matrix

from discriminator.data_set_reader import create_readers
import utils
import discriminator.model as model
import args


class Experiment:

    def __init__(self, prefix_checkpoint, is_testing=False, num_channels=2, learning_rate_init=10 ** -3, num_epochs=30, batch_size=64):
        tf.reset_default_graph()
        self.prefix = prefix_checkpoint
        self.learning_rate_init = learning_rate_init
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        # self.checkpoint_folder = os.path.join(args.CHECKPOINTS_BASE, "network_disc_latent_%s" % prefix_checkpoint)
        if prefix_checkpoint.find('app') != -1:
            self.checkpoint_folder = os.path.join(args.CHECKPOINTS_BASE, "network_disc_diff_mask_%s" % prefix_checkpoint)
        else:
            self.checkpoint_folder = os.path.join(args.CHECKPOINTS_BASE, "network_disc_latent_%s" % prefix_checkpoint)
        self.IS_RESTORE = tf.train.latest_checkpoint(self.checkpoint_folder) is not None
        self.inputs_ = tf.placeholder(np.float32, [None, 64, 64, num_channels])
        self.is_training = tf.placeholder(np.bool, None)
        self.targets_ = tf.placeholder(np.float32, [None, 1])

        # build neural network
        self.logits = model.lenet(self.inputs_, self.is_training)
        self.cost = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.targets_)
        self.avg_cost = tf.reduce_mean(self.cost)
        self.global_step = tf.Variable(0, trainable=False)  # 782
        self.learning_rate = learning_rate_init
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.avg_cost,
                                                                                           global_step=self.global_step)
        self.sess = tf.Session(config=args.tf_config)

        self.train_loss_placeholder = tf.placeholder(tf.float32, shape=[], name="train_loss")
        self.val_loss_placeholder = tf.placeholder(tf.float32, shape=[], name="val_loss")

        self.train_acc_placeholder = tf.placeholder(tf.float32, shape=[], name="train_acc")
        self.val_acc_placeholder = tf.placeholder(tf.float32, shape=[], name="val_acc")

        tf.summary.scalar('train_loss', self.train_loss_placeholder)
        tf.summary.scalar('val_loss', self.val_loss_placeholder)

        tf.summary.scalar('train_acc', self.train_acc_placeholder)
        tf.summary.scalar('val_acc', self.val_acc_placeholder)
        tf.summary.scalar('learning_rate', self.learning_rate)
        self.merged = tf.summary.merge_all()
        if is_testing:
            self.restore_model()

    def predict(self, image):
        
        if self.prefix.find('app') != -1:
            images = np.expand_dims(np.expand_dims(image, axis=2), axis=0)
        else:
            images = np.expand_dims(image, axis=0)

        predictions = self.sess.run(self.logits, feed_dict={self.inputs_: images, self.is_training: False})
        predictions = utils.sigmoid(predictions)
        return predictions[0, 0]

    def fit(self, reader_train):
        iters = int(np.ceil(reader_train.num_samples / self.batch_size))
        total_loss = 0
        un_norm_acc = 0
        for iter in range(iters):
            batch_x, batch_y = reader_train.next_batch(self.batch_size)
            _, c, predictions, _ = self.sess.run([self.optimizer, self.cost, self.logits, self.global_step],
                    feed_dict={self.inputs_: batch_x,  self.targets_: batch_y, self.is_training: True})
            total_loss += np.sum(c)
            predictions = (predictions > 0) * 1
            un_norm_acc += np.sum(np.round(predictions) == batch_y)
            # pdb.set_trace()
        
        return total_loss / reader_train.num_samples, un_norm_acc / reader_train.num_samples

    def eval(self, reader, return_predicted_labels=False):
        iters = int(np.ceil(reader.num_samples / self.batch_size))
        total_loss = 0
        un_norm_acc = 0
        if return_predicted_labels:
            pred_labels = None

        for iter in range(iters):
            batch_x, batch_y = reader.next_batch(self.batch_size)
            c, predictions = self.sess.run([self.cost, self.logits],
                    feed_dict={self.inputs_: batch_x,  self.targets_: batch_y, self.is_training: False})
            total_loss += np.sum(c)
            predictions = (predictions > 0) * 1            
            un_norm_acc += np.sum(predictions == batch_y)
            # pdb.set_trace()
            # if return_predicted_labels:
            #     if pred_labels is None:
            #         pred_labels = np.argmax(predictions, axis=1)
            #     else:
            #         pred_labels = np.concatenate((pred_labels, np.argmax(predictions, axis=1)))

        if return_predicted_labels:
            return total_loss / reader.num_samples, un_norm_acc / reader.num_samples, pred_labels
        else:
            return total_loss / reader.num_samples, un_norm_acc / reader.num_samples

    def restore_model(self, epoch=None):
        if epoch is None:
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_folder)
        else:
            checkpoint_path = os.path.join(self.checkpoint_folder, "model_%d" % epoch)
        # pdb.set_trace()
        if checkpoint_path is None:
            raise Exception("Checkpoint file is missing!")

        self.sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=0)
        saver.restore(self.sess, checkpoint_path)

    def get_statistics_set(self, reader, epoch=None):
        self.restore_model(epoch=epoch)
        loss, acc, predicted_labels = self.eval(reader, return_predicted_labels=True)
        utils.log_message('loss = {}, acc = {} \nconf mat = \n{}'.format(loss, acc, confusion_matrix(np.argmax(reader.labels, axis=1), predicted_labels)))

    # def get_statistics(self, epoch=None):
    #     utils.log_message("Statistics for epoch: {}".format(epoch))
    #     utils.log_message("TRAINING")
    #     self.get_statistics_set(reader_train, epoch)
    #     utils.log_message("VAL")
    #     self.get_statistics_set(reader_val, epoch)

    def run(self, reader_train, reader_val):
        start_epoch = 0
        saver = tf.train.Saver(max_to_keep=0)
        self.sess.run(tf.global_variables_initializer())
        if self.IS_RESTORE:
            print('=' * 30 + '\nRestoring from ' + tf.train.latest_checkpoint(self.checkpoint_folder))
            saver.restore(self.sess, tf.train.latest_checkpoint(self.checkpoint_folder))
            start_epoch = re.findall(r'\d+', tf.train.latest_checkpoint(self.checkpoint_folder))
            start_epoch = int(start_epoch[-1]) + 1

        writer = tf.summary.FileWriter(os.path.join(args.CHECKPOINTS_BASE, 'train_disc.log'), self.sess.graph)

        for epoch in range(start_epoch, self.num_epochs):
            utils.log_message("Epoch: %d/%d" % (epoch, self.num_epochs))
            train_loss, train_acc = self.fit(reader_train)
            val_loss, val_acc = self.eval(reader_val)
            utils.log_message("acc train = {}, val = {}.".format(train_acc, val_acc))
            utils.log_message("loss train = %.4f, val = %.4f" % (train_loss, val_loss))
            merged_ = self.sess.run(self.merged, feed_dict={
                                                            self.train_loss_placeholder: train_loss,
                                                            self.val_loss_placeholder: val_loss,
                                                            self.train_acc_placeholder: train_acc,
                                                            self.val_acc_placeholder: val_acc})
            writer.add_summary(merged_, epoch)

            saver.save(self.sess, os.path.join(self.checkpoint_folder, "model_%d" % epoch))


def train(prefix_ae):
    # readers are created here
    if prefix_ae.find('app') != -1:
        num_channels = 8
    else:
        num_channels = 16
    
    reader_train, reader_val = create_readers(prefix_ae, num_channels=num_channels)
    exp = Experiment(prefix_ae, num_channels=num_channels)
    exp.run(reader_train, reader_val)
