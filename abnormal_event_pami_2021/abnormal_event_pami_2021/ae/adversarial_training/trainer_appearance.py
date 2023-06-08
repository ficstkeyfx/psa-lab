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
from utils import ProcessingType, create_dir, log_message, log_function_start, log_function_end, concat_images
import ae.adversarial_training.conv_autoencoder as cae
from ae.adversarial_training.dataset_reader_adversarial import *

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
        self.targets_ = tf.placeholder(tf.float32, (None, self.input_size[0], self.input_size[1], 1), name='targets')
        self.target_masks = tf.placeholder(tf.float32, (None, self.input_size[0], self.input_size[1], 1), name='masks')
        self.session = tf.Session(config=args.tf_config)
        self.encoded, self.conv1, self.conv2, self.conv3 = cae.encoder(self.inputs_)

        self.encoder_variables = tf.global_variables()
        self.decoded_target = cae.decoder(self.encoded, "target", 1, self.conv1, self.conv2, self.conv3)

        self.decoded_adversarial = cae.decoder(self.encoded, "adversarial", 1, self.conv1, self.conv2, self.conv3)
        all_variables = tf.global_variables()
        self.adversarial_variables = [var for var in all_variables if var.name.find("adversarial") != -1]

        self.masks = cae.decoder_mask(self.encoded, "mask")
        self.cost_masks = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.target_masks, logits=self.masks)

        self.cost_target = tf.square(self.decoded_target - self.targets_)
        self.cost_target = tf.multiply(self.cost_target, self.target_masks)
        self.loss_target = tf.reduce_mean(self.cost_target) + tf.reduce_mean(self.cost_masks)
        self.cost_adversarial = tf.square(self.decoded_adversarial - self.targets_)
        self.loss_adversarial = tf.reduce_mean(self.cost_adversarial)

        self.__is_session_initialized = False
        self.IS_RESTORE = tf.train.latest_checkpoint(self.checkpoint_folder) is not None
        self.name = "ae_appearance"

    def restore_model(self, epoch=None):
        if epoch is None:
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_folder)
        else:
            checkpoint_path = os.path.join(self.checkpoint_folder, "ae_model_%d" % epoch)
        # pdb.set_trace()
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
        decoded_, masks_ = self.session.run([self.decoded_target, self.masks], feed_dict={self.inputs_: images})
        # decoded_ = self.session.run(self.decoded_target, feed_dict={self.inputs_: images})
        masks_[masks_ < 0] = 0
        masks_[masks_ > 0] = 1
        decoded_ = np.uint8(np.clip(np.round(decoded_ * 255.0), 0, 255))
        
        return decoded_, masks_


    def get_latent_features(self, images):
        self.__check_session()
        encoded_ = self.session.run(self.encoded, feed_dict={self.inputs_: images})
        return encoded_

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, data_reader: DataSetReaderAppearance, data_reader_adversarial: DataSetReaderAdversarial,
              learning_rate=10 ** -4,
              num_epochs=20,
              batch_size=64):

        log_function_start()
        opt_target = tf.train.AdamOptimizer(learning_rate).minimize(self.loss_target)
        # optimizer adversarial
        alpha = -args.lambda_
        assert alpha <= 0
        opt1 = tf.train.AdamOptimizer(learning_rate * alpha)
        opt2 = tf.train.AdamOptimizer(learning_rate)
        grads = tf.gradients(self.loss_adversarial, self.encoder_variables + self.adversarial_variables)
        grads1 = grads[:len(self.encoder_variables)]
        grads2 = grads[len(self.encoder_variables):]
        train_op1 = opt1.apply_gradients(zip(grads1, self.encoder_variables))
        train_op2 = opt2.apply_gradients(zip(grads2, self.adversarial_variables))
        opt_adversarial = tf.group(train_op1, train_op2)
        # end opt adversarial

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
        tf.summary.scalar('loss target on batch', self.loss_target)
        tf.summary.scalar('loss adversarial on batch', self.loss_adversarial)
        tf.summary.scalar('learning rate', learning_rate)
        merged = tf.summary.merge_all()
        iterations = int(math.ceil(float(data_reader.num_images) / batch_size))
        log_message("Number of train images: %d" % data_reader.num_images)

        for epoch in range(start_epoch, num_epochs):
            log_message("Epoch: %d/%d" % (epoch, num_epochs))
            for iteration in range(0, iterations):
                batch_input, batch_target, batch_masks = data_reader.get_next_batch(iteration, batch_size)
                batch_loss, _, decoded_target, encoded_, my_mask = self.session.run([self.loss_target, opt_target,
                                                                            self.decoded_target,
                                                                            self.encoded, self.masks],
                                                                      feed_dict={self.inputs_: batch_input,
                                                                                 self.targets_: batch_target,
                                                                                 self.target_masks: batch_masks
                                                                                 })
 
                # mm = self.sigmoid(my_mask[0])
                # cv.imshow('pred', np.uint8(mm * 255))
                # cv.imshow('gt', np.uint8(batch_masks[0] * 255))
                # cv.imshow('gtr', np.uint8(batch_input[0] * 255))
                # cv.imshow('predd', np.uint8(decoded_target[0] * 255))
                # cv.waitKey(0)

                # print("Epoch: {}/{} iteration: {}/{}...".format(epoch, num_epochs, iteration, iterations),
                #       "Training loss target: {:.4f}".format(batch_loss))
                batch_input_adversarial, batch_target_adversarial = data_reader_adversarial.get_next_batch(iteration, batch_size)
                batch_loss, _, decoded_, encoded_ = self.session.run([self.loss_adversarial, opt_adversarial, self.decoded_adversarial,
                                                                               self.encoded],
                                                                      feed_dict={self.inputs_: batch_input_adversarial,
                                                                          self.targets_: batch_target_adversarial})

                # writer.add_summary(merged_, epoch * iterations + iteration)

                # print("Epoch: {}/{} iteration: {}/{}...".format(epoch, num_epochs, iteration, iterations),
                #      "Training loss adversarial: {:.4f}".format(batch_loss))

            print('Saving checkpoint...', epoch)
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
                file_path = file_path.replace(args.samples_folder_name, "appearance_latent_features_unet_%f" % args.lambda_)
                file_path = file_path.replace("_01.png", ".npy")
                dir_name, file_name = os.path.split(file_path)
                create_dir(dir_name)
                np.save(file_path, encoded_[idx].flatten())
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
            decoded_, _ = self.get_reconstructed_images(batch_input)
            for idx, file_path in enumerate(file_paths):
                current_image = decoded_[idx]
                if write_to_disk is True:

                    file_path = file_path.replace(args.samples_folder_name, "reconstructed_images_appearance")
                    dir_name, file_name = os.path.split(file_path)
                    
                    if dir_name == '':
                        file_path = os.path.join(args.output_folder_base, "reconstructed_images_appearance_adv_unet", file_path)
                        dir_name, file_name = os.path.split(file_path)
                    create_dir(dir_name)
                    img_to_save = concat_images(current_image, np.uint8(np.round(batch_input[idx] * 255.0)))
                    cv.imwrite(file_path, img_to_save)
                else:
                    cv.imshow("original image", np.uint8(np.round(batch_input[idx] * 255.0)))
                    cv.imshow("reconstructed image", current_image)
                    cv.waitKey(1000)
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
            decoded_, masks_ = self.get_reconstructed_images(batch_input)
            for idx, file_path in enumerate(file_paths):
                file_path = file_path.replace(args.samples_folder_name,
                                              "appearance_reconstruction_features_unet_%f" % args.lambda_)
                file_path = file_path.replace("_01.png", ".npy")
                dir_name, file_name = os.path.split(file_path)
                create_dir(dir_name)
                np.save(file_path, decoded_[idx].flatten())
                # file_path = file_path.replace(".npy", "_mask.npy")
                # np.save(file_path, masks_[idx].flatten())
        log_function_end()

    def compute_reconstruction_features_for_ae(self, data_reader_, is_adv=False,
                                        epoch=None,
                                        batch_size=64):
        log_function_start()
        self.restore_model(epoch)
        if is_adv:
            dir_name = os.path.join(args.output_folder_base, args.CHECKPOINTS_PREFIX, "app_diff_adv")
        else:
            dir_name = os.path.join(args.output_folder_base, args.CHECKPOINTS_PREFIX, "app_diff_target")
        create_dir(dir_name)
        log_message("Number of images: %d" % data_reader_.num_images)
        iterations = int(math.ceil(float(data_reader_.num_images) / batch_size))
        for iteration in range(0, iterations):
            batch_input, batch_target, file_paths = data_reader_.get_next_batch(iteration, batch_size, add_noise=False,
                                                                                return_file_names=True)
            decoded_, masks_  = self.get_reconstructed_images(batch_input)
            for idx, file_path in enumerate(file_paths):
                split_name = file_path.split(os.sep)
                if is_adv is True:
                    name = file_path[:-4] + '.npy'
                else:
                    name = split_name[-3] + "_" + split_name[-1][:-4] + '.npy'

                dst_file = os.path.join(dir_name, name)
                diff = np.abs(batch_target[idx] - (decoded_[idx] / 255))

                np.save(dst_file, diff.flatten())

        log_function_end()

    def compute_latent_features_for_ae(self, data_reader_, is_adv=False, epoch=None,  batch_size=64):
        log_function_start()
        self.restore_model(epoch)
        if is_adv:
            dir_name = os.path.join(args.output_folder_base, args.CHECKPOINTS_PREFIX, "app_latent_adv")
        else:
            dir_name = os.path.join(args.output_folder_base, args.CHECKPOINTS_PREFIX, "app_latent_target")
        create_dir(dir_name)
        log_message("Number of images: %d" % data_reader_.num_images)
        iterations = int(math.ceil(float(data_reader_.num_images) / batch_size))
        for iteration in range(0, iterations):
            batch_input, batch_target, file_paths = data_reader_.get_next_batch(iteration, batch_size, add_noise=False,
                                                                                return_file_names=True)
            encoded_ = self.get_latent_features(batch_input)
            for idx, file_path in enumerate(file_paths):
                split_name = file_path.split(os.sep)
                if is_adv is True:
                    name = file_path[:-4] + '.npy'
                else:
                    name = split_name[-3] + "_" + split_name[-1][:-4] + '.npy'
                dst_file = os.path.join(dir_name, name)
                np.save(dst_file, encoded_[idx].flatten())

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
                                                                            max_bbox_size=max_bbox_size,
                                                                            is_testing=False)

    data_reader_training_adversarial = DataSetReaderAdversarial(os.path.join(args.adversarial_images_path),
                                                                input_size=ae_trainer.input_size)

    ae_trainer.train(data_reader_training, data_reader_training_adversarial)
    ae_trainer.compute_reconstruction_features_for_ae(data_reader_training, False)
    ae_trainer.compute_reconstruction_features_for_ae(data_reader_training_adversarial, True)
   
    ae_trainer.compute_latent_features_for_ae(data_reader_training, False)
    ae_trainer.compute_latent_features_for_ae(data_reader_training_adversarial, True)
    
    # ae_trainer.compute_latent_features(data_reader_training)
    # ae_trainer.compute_reconstruction_features(data_reader_training)
    # ae_trainer.visualise_reconstructed_images(data_reader_training, write_to_disk=True)
    # ae_trainer.visualise_reconstructed_images(data_reader_training_adversarial, write_to_disk=True)
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
    ae_trainer.compute_latent_features(data_reader_test)
    # ae_trainer.visualise_reconstructed_images(data_reader_test)
    ae_trainer.compute_reconstruction_features(data_reader_test)
    ae_trainer.close_session()
    log_function_end()

