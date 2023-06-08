#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import tensorflow as tf
import re
import math
import cv2 as cv
import pdb

import args
from dataset_reader_appearance import *
from utils import ProcessingType, create_dir, log_message, log_function_start, log_function_end, concat_images, sigmoid
import ae.conv_autoencoder as cae

min_bbox_size = 0
max_bbox_size = 300


class AppearanceAe:

    def __init__(self, append_to_path=None):
        # define placeholders
        self.checkpoint_folder = os.path.join(args.CHECKPOINTS_BASE, "ae_appearance")
        self.append_to_path = append_to_path
        if append_to_path is not None:
            self.checkpoint_folder = os.path.join(append_to_path, self.checkpoint_folder)

        self.input_size = (64, 64)
        self.inputs_ = tf.placeholder(tf.float32, (None, self.input_size[0], self.input_size[1], 1), name='inputs')
        # self.target_masks = tf.placeholder(tf.float32, (None, self.input_size[0], self.input_size[1], 1), name='target_masks')
        self.targets_ = tf.placeholder(tf.float32, (None, self.input_size[0], self.input_size[1], 1), name='targets')
        self.session = tf.Session(config=args.tf_config)
        self.decoded, self.encoded = cae.model(self.inputs_)
        # self.masks = cae.decoder(self.encoded)
        self.cost = tf.square(self.decoded - self.targets_)
        # self.cost = tf.multiply(self.cost, self.target_masks)
        # self.cost_masks = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_masks, logits=self.masks)
        self.loss = tf.reduce_mean(self.cost) # + tf.reduce_mean(self.cost_masks)
        self.__is_session_initialized = False
        self.IS_RESTORE = tf.train.latest_checkpoint(self.checkpoint_folder) is not None
        self.name = "ae_appearance"

    def restore_model(self, epoch=None):
        if epoch is None:
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_folder)
        else:
            checkpoint_path = os.path.join(self.checkpoint_folder, "ae_model_%d" % epoch)

        if checkpoint_path is None:
            raise Exception("Checkpoint file is missing!")

        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=0)
        saver.restore(self.session, checkpoint_path)
        self.__is_session_initialized = True

    def __check_session(self):
        if self.__is_session_initialized is False:
            raise Exception("Session is not initialized!")
        return True

    def get_reconstructed_images(self, images):
        self.__check_session()
        decoded_ = self.session.run(self.decoded, feed_dict={self.inputs_: images})
        
        # masks_[masks_ < 0] = 0
        # masks_[masks_ > 0] = 1
        decoded_ = np.uint8(np.clip(np.round(decoded_ * 255.0), 0, 255))
        return decoded_ # , masks_
    """
       def get_reconstructed_images(self, images):
                   self.__check_session()
                           decoded_, masks_ = self.session.run([self.decoded, self.masks], feed_dict={self.inputs_: images})

                                   masks_[masks_ < 0] = 0
                                           masks_[masks_ > 0] = 1
                                                   decoded_ = np.uint8(np.clip(np.round(decoded_ * 255.0), 0, 255))
                                                           return decoded_, masks_
                                                       """

    def get_latent_features(self, images):
        self.__check_session()
        encoded_ = self.session.run(self.encoded, feed_dict={self.inputs_: images})
        return encoded_

    def train(self, data_reader: DataSetReaderAppearance,
              learning_rate=10 ** -4,
              num_epochs=30,
              batch_size=64):

        log_function_start()
        opt = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=0)
        start_epoch = 0
        if self.IS_RESTORE:
            log_message('=' * 30 + '\nRestoring from ' + tf.train.latest_checkpoint(self.checkpoint_folder))
            saver.restore(self.session, tf.train.latest_checkpoint(self.checkpoint_folder))
            start_epoch = re.findall(r'\d+', tf.train.latest_checkpoint(self.checkpoint_folder))
            start_epoch = int(start_epoch[-1]) + 1

        # summary section
        writer = tf.summary.FileWriter(os.path.join(args.CHECKPOINTS_BASE, 'train_appearance.log'), self.session.graph)
        tf.summary.scalar('loss on batch', self.loss)
        tf.summary.scalar('learning rate', learning_rate)
        merged = tf.summary.merge_all()
        iterations = int(math.ceil(float(data_reader.num_images) / batch_size))
        log_message("Number of train images: %d" % data_reader.num_images)

        for epoch in range(start_epoch, num_epochs):
            log_message("Epoch: %d/%d" % (epoch, num_epochs))
            for iteration in range(0, iterations):
                batch_input, batch_target, target_masks = data_reader.get_next_batch(iteration, batch_size)
                batch_loss, _, merged_, decoded_, encoded_ = self.session.run([self.loss, opt, merged, self.decoded,
                                                                               self.encoded],
                                                                      feed_dict={self.inputs_: batch_input,
                                                                                 self.targets_: batch_target})

                # mm = sigmoid(my_masks[0])
                # mm[mm > 0.5] = 1
                # mm[mm <= 0.5] = 0
                # cv.imshow('pred', np.uint8(mm * 255))
                # cv.imshow('gt', np.uint8(target_masks[0] * 255))
                # cv.imshow('gtr', np.uint8(batch_input[0] * 255))
                # cv.imshow('predd', np.uint8(decoded_[0] * 255))
                # cv.waitKey(0)
                # cv.imshow('a', np.uint8(decoded_[0] * 255)); cv.waitKey(100);
                writer.add_summary(merged_, epoch * iterations + iteration)

                print("Epoch: {}/{} iteration: {}/{}...".format(epoch, num_epochs, iteration, iterations),
                      "Training loss: {:.4f}".format(batch_loss))

            print('Saving checkpoint...')
            saver.save(self.session, os.path.join(self.checkpoint_folder, "ae_model_%d" % epoch))

        log_function_end()

    def compute_latent_features(self, data_reader_,
                                epoch=None,
                                batch_size=64):
        log_function_start()
        self.restore_model(epoch)

        log_message("Number of images: %d" % data_reader_.num_images)
        iterations = int(math.ceil(float(data_reader_.num_images) / batch_size))
        for iteration in range(0, iterations):
            batch_input, batch_target, file_paths = data_reader_.get_next_batch(iteration, batch_size, add_noise=False,
                                                                                return_file_names=True)
            encoded_ = self.get_latent_features(batch_input)
            for idx, file_path in enumerate(file_paths):
                file_path = file_path.replace(args.samples_folder_name, "appearance_latent_features")
                file_path = file_path.replace("_01.png", ".txt")
                dir_name, file_name = os.path.split(file_path)
                create_dir(dir_name)
                np.savetxt(file_path, encoded_[idx].flatten())
        log_function_end()

    def compute_reconstruction_features(self, data_reader_,
                                        epoch=None,
                                        batch_size=64):
        log_function_start()
        self.restore_model(epoch)

        log_message("Number of images: %d" % data_reader_.num_images)
        iterations = int(math.ceil(float(data_reader_.num_images) / batch_size))
        for iteration in range(0, iterations):
            batch_input, batch_target, file_paths = data_reader_.get_next_batch(iteration, batch_size, add_noise=False,
                                                                                return_file_names=True)
            decoded_  = self.get_reconstructed_images(batch_input)
            for idx, file_path in enumerate(file_paths):
                file_path = file_path.replace(args.samples_folder_name, "appearance_reconstruction_features")
                file_path = file_path.replace("_01.png", ".npy")
                dir_name, file_name = os.path.split(file_path)
                create_dir(dir_name)
                np.save(file_path, decoded_[idx].flatten())
                # file_path = file_path.replace(".npy", "_mask.npy")
                # np.save(file_path, masks_[idx].flatten())
        log_function_end()

    def visualise_reconstructed_images(self, data_reader_,
                                       write_to_disk=True,
                                       epoch=None,
                                       batch_size=64):
        log_function_start()

        self.restore_model(epoch)
        log_message("Number of images: %d" % data_reader_.num_images)
        iterations = int(
            math.ceil(float(min(data_reader_.num_images, args.num_samples_for_visualization)) / batch_size))

        for iteration in range(0, iterations):
            batch_input, batch_target, file_paths = data_reader_.get_next_batch(iteration, batch_size, add_noise=False,
                                                                                return_file_names=True)
            decoded_ = self.get_reconstructed_images(batch_input)
            for idx, file_path in enumerate(file_paths):
                current_image = decoded_[idx]
                if write_to_disk is True:
                    file_path = file_path.replace(args.samples_folder_name, "reconstructed_images_appearance")
                    dir_name, file_name = os.path.split(file_path)
                    create_dir(dir_name)
                    img_to_save = concat_images(current_image, np.uint8(np.round(batch_input[idx] * 255.0)))
                    cv.imwrite(file_path, img_to_save)
                else:
                    cv.imshow("original image", np.uint8(np.round(batch_input[idx] * 255.0)))
                    cv.imshow("reconstructed image", current_image)
                    cv.waitKey(1000)
        log_function_end()

    def close_session(self):
        self.session.close()
        self.__is_session_initialized = False


def train():
    log_function_start()
    tf.reset_default_graph()
    ae_trainer = AppearanceAe()
    data_reader_training: DataSetReaderAppearance = DataSetReaderAppearance(os.path.join(args.output_folder_base,
                                                                                         args.database_name,
                                                                                         ProcessingType.TRAIN.value),
                                                                            args.samples_folder_name,
                                                                            input_size=ae_trainer.input_size,
                                                                            min_bbox_size=min_bbox_size,
                                                                            max_bbox_size=max_bbox_size)

    ae_trainer.train(data_reader_training)
    # ae_trainer.compute_latent_features(data_reader_training)
    ae_trainer.visualise_reconstructed_images(data_reader_training, write_to_disk=True)
    ae_trainer.close_session()
    log_function_end()


def test():
    log_function_start()
    tf.reset_default_graph()
    ae_trainer = AppearanceAe()
    data_reader_test: DataSetReaderAppearance = DataSetReaderAppearance(os.path.join(args.output_folder_base,
                                                                                     args.database_name,
                                                                                     ProcessingType.TEST.value),
                                                                        args.samples_folder_name,
                                                                        input_size=ae_trainer.input_size,
                                                                        min_bbox_size=min_bbox_size,
                                                                        max_bbox_size=max_bbox_size, is_testing=True)
    # ae_trainer.compute_latent_features(data_reader_test)
    ae_trainer.compute_reconstruction_features(data_reader_test)
    ae_trainer.visualise_reconstructed_images(data_reader_test)
    ae_trainer.close_session()
    log_function_end()

