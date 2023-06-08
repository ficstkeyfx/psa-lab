#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import tensorflow as tf
import re
import math
import cv2 as cv
import pdb
from scipy import stats

import args
from dataset_reader_motion_optical_flow import *
from ae.adversarial_training.dataset_reader_adversarial_motion import *
from utils import concat_images, ProcessingType, create_dir, log_message, log_function_start, log_function_end
import ae.adversarial_training.conv_autoencoder as cae


min_bbox_size = 0
max_bbox_size = 300


class MotionAe:
    input_size = (64, 64)

    def __init__(self, ae_motion_type: MotionAeType, append_to_path=None):

        self.ae_motion_type = ae_motion_type
        self.checkpoint_folder = os.path.join(args.CHECKPOINTS_BASE, "ae_motion_" + ae_motion_type.value)
        self.IS_RESTORE = tf.train.latest_checkpoint(self.checkpoint_folder) is not None

        self.append_to_path = append_to_path
        if append_to_path is not None:
            self.checkpoint_folder = os.path.join(append_to_path, self.checkpoint_folder)

        self.inputs_ = tf.placeholder(tf.float32, (None, self.input_size[0], self.input_size[1], 2), name='inputs')
        self.targets_ = tf.placeholder(tf.float32, (None, self.input_size[0], self.input_size[1], 2), name='targets')

        self.session = tf.Session(config=args.tf_config)
        self.encoded, self.conv1, self.conv2, self.conv3 = cae.encoder(self.inputs_)

        self.encoder_variables = tf.global_variables()
        self.decoded_target = cae.decoder(self.encoded, "target", 2, self.conv1, self.conv2, self.conv3)
        self.decoded_adversarial = cae.decoder(self.encoded, "adversarial", 2, self.conv1, self.conv2, self.conv3)

        all_variables = tf.global_variables()
        self.adversarial_variables = [var for var in all_variables if var.name.find("adversarial") != -1]
        self.cost_target = tf.square(self.decoded_target - self.targets_)
        self.loss_target = tf.reduce_mean(self.cost_target)

        self.cost_adversarial = tf.square(self.decoded_adversarial - self.targets_)
        self.loss_adversarial = tf.reduce_mean(self.cost_adversarial)

        self.__is_session_initialized = False
        self.name = "ae_motion_" + ae_motion_type.value

    def restore_model(self, epoch=None):
        if epoch is None:
            checkpoint_path = tf.train.latest_checkpoint(self.checkpoint_folder)
        else:
            checkpoint_path = os.path.join(self.checkpoint_folder, "ae_model_%d" % epoch)

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
        decoded_ = self.session.run(self.decoded_target, feed_dict={self.inputs_: images})
        decoded_ = np.uint8(np.clip(np.round(decoded_ * 255.0), 0, 255))
        return decoded_

    def get_reconstructed_flow(self, images):
        self.__check_session()
        decoded_ = self.session.run(self.decoded_target, feed_dict={self.inputs_: images})
        return decoded_

    def get_latent_features_feed_dict(self, images):
        self.__check_session()
        encoded_ = self.session.run(self.encoded, feed_dict={self.inputs_: images})
        return encoded_

    def get_latent_features(self):
        self.__check_session()
        encoded_, names = self.session.run([self.encoded, self.names_])
        return encoded_, names

    def close_session(self):
        self.session.close()
        self.__is_session_initialized = False

    def train(self, data_reader, data_reader_adversarial,
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
            print('=' * 30 + '\nRestoring from ' + tf.train.latest_checkpoint(self.checkpoint_folder))
            saver.restore(self.session, tf.train.latest_checkpoint(self.checkpoint_folder))
            start_epoch = re.findall(r'\d+', tf.train.latest_checkpoint(self.checkpoint_folder))
            start_epoch = int(start_epoch[0]) + 1

        # summary section
        writer = tf.summary.FileWriter(os.path.join(args.CHECKPOINTS_BASE,
                                                    'train_motion_%s.log' % self.ae_motion_type.value), self.session.graph)
        # tf.summary.scalar('loss on batch', self.loss)
        # tf.summary.scalar('learning rate', learning_rate)
        merged = tf.summary.merge_all()

        log_message("Number of train images: %d" % data_reader.num_images)
        for epoch in range(start_epoch, num_epochs):
            log_message("Epoch: %d/%d" % (epoch, num_epochs))
            iterations = int(math.ceil(float(data_reader.num_images) / batch_size))

            for iteration in range(0, iterations):
                # print(iteration)
                batch_input, batch_target = data_reader.get_next_batch(iteration, batch_size)
                batch_loss, _, decoded_target, encoded_ = self.session.run([self.loss_target, opt_target,
                                                                                     self.decoded_target,
                                                                                     self.encoded],
                                                                                    feed_dict={
                                                                                        self.inputs_: batch_input,
                                                                                        self.targets_: batch_target})

                # print("Epoch: {}/{} iteration: {}/{}...".format(epoch, num_epochs, iteration, iterations),
                #      "Training loss target: {:.4f}".format(batch_loss))

                batch_input_adversarial, batch_target_adversarial = data_reader_adversarial.get_next_batch(iteration, batch_size)
                batch_loss, _, decoded_, encoded_ = self.session.run([self.loss_adversarial, opt_adversarial, self.decoded_adversarial,
                                                                               self.encoded],
                                                                     feed_dict={self.inputs_: batch_input_adversarial,
                                                                                self.targets_: batch_target_adversarial})

               # print("Epoch: {}/{} iteration: {}/{}...".format(epoch, num_epochs, iteration, iterations),
               #       "Training loss adversarial: {:.4f}".format(batch_loss))

            print('Saving checkpoint...')
            saver.save(self.session, os.path.join(self.checkpoint_folder, "ae_model_%d" % epoch))

        log_function_end()

    def compute_latent_features_feed_dict(self, data_reader_,
                                          epoch=None,
                                          batch_size=64):
        log_function_start()
        self.restore_model(epoch)

        log_message("Number of images: %d" % data_reader_.num_images)
        iterations = int(math.ceil(float(data_reader_.num_images) / batch_size))
        for iteration in range(0, iterations):
            batch, _, file_paths = data_reader_.get_next_batch(iteration, batch_size, return_file_names=True)
            encoded_ = self.get_latent_features_feed_dict(batch)
            for idx, file_path in enumerate(file_paths):
                file_path = file_path.replace(data_reader_.folder_name,
                                              "motion_latent_features_unet_%f_" % args.lambda_ +
                                              data_reader_.ae_type.value)
                file_path = file_path.replace("_01.png", ".npy")
                dir_name, file_name = os.path.split(file_path)
                create_dir(dir_name)
                np.save(file_path, encoded_[idx].flatten())
        log_function_end()

    def compute_latent_features(self, data_reader_,
                                batch_size, epoch=None):
        log_function_start()
        self.restore_model(epoch)

        log_message("Number of images: %d" % data_reader_.num_images)
        iterations = int(math.ceil(float(data_reader_.num_images) / batch_size))
        self.session.run(self.iter.initializer)  # reinitialize the iterator

        for iteration in range(0, iterations):
            encoded_, file_paths = self.get_latent_features()
            # print(file_paths)
            for idx, file_path in enumerate(file_paths):
                file_path = file_path.decode().replace(data_reader_.folder_name, "motion_latent_features_unet_concat_" +
                                                       data_reader_.ae_type.value)
                
                dir_name, file_name = os.path.split(file_path)
                create_dir(dir_name)
                np.save(file_path, encoded_[idx].flatten())
        log_function_end()

    def compute_reconstruction_features(self, data_reader_,
                                            batch_size, epoch=None):
        log_function_start()
        self.restore_model(epoch)

        log_message("Number of images: %d" % data_reader_.num_images)
        iterations = int(math.ceil(float(data_reader_.num_images) / batch_size))
        for iteration in range(0, iterations):
            batch, batch_copy, file_paths = data_reader_.get_next_batch(iteration, batch_size, return_file_names=True)
            decoded_ = self.get_reconstructed_flow(batch)
            for idx, file_path in enumerate(file_paths):
                
                file_path = file_path.replace(data_reader_.folder_name,
                                              "motion_reconstruction_features_unet_%f_" % args.lambda_ +
                                              data_reader_.ae_type.value)

                dir_name, file_name = os.path.split(file_path)
                create_dir(dir_name)
                np.save(file_path, decoded_[idx].flatten())
        log_function_end()

    def compute_max_error(self, data_reader_, batch_size=64, epoch=None):
        log_function_start()
        self.restore_model(epoch)
        errs = []
        log_message("Number of images: %d" % data_reader_.num_images)
        iterations = int(math.ceil(float(data_reader_.num_images) / batch_size))
        for iteration in range(0, iterations):
            batch, batch_copy, file_paths = data_reader_.get_next_batch(iteration, batch_size, return_file_names=True)
            decoded_ = self.get_reconstructed_flow(batch)
            for idx, file_path in enumerate(file_paths):
                res = np.mean(np.abs(batch[idx] - decoded_[idx]))
                errs.append(res)
                # pdb.set_trace()

        errs = np.array(errs)
        print('mean error', np.mean(errs))
        print('min error', np.min(errs))
        print('max error', np.max(errs))
        print('mediana', stats.mode(errs))
        np.save('errors_motion.npy', errs)
        log_function_end()

    def visualise_reconstructed_images(self, data_reader_,
                                       write_to_disk=True,
                                       epoch=None,
                                       batch_size=64):
        log_function_start()
        self.restore_model(epoch)
        self.session.run(self.iter.initializer)
        log_message("Number of images: %d" % data_reader_.num_images)
        iterations = int(
            math.ceil(float(min(data_reader_.num_images, args.num_samples_for_visualization)) / batch_size))
        for iteration in range(0, iterations):
            batch_input, batch_target, file_paths = data_reader_.get_next_batch(iteration, batch_size,
                                                                                return_file_names=True)
            decoded_ = self.get_reconstructed_images(batch_input)
            for idx, file_path in enumerate(file_paths):
                current_image = decoded_[idx]
                if write_to_disk is True:
                    file_path = file_path.replace(args.samples_folder_name, "reconstructed_images_"
                                                  + data_reader_.ae_type.value)
                    dir_name, file_name = os.path.split(file_path)
                    create_dir(dir_name)
                    img_to_save = concat_images(current_image, np.uint8(np.round(batch_input[idx] * 255.0)))
                    cv.imwrite(file_path, img_to_save)
                else:
                    cv.imshow("original image", np.uint8(np.round(batch_input[idx] * 255.0)))
                    cv.imshow("reconstructed image", current_image)
                    cv.waitKey(1000)
        log_function_end()

    def compute_reconstruction_features_for_ae(self, data_reader_, is_adv=False,
                                        epoch=None,
                                        batch_size=64):
        log_function_start()
        self.restore_model(epoch)
        if is_adv:
            dir_name = os.path.join(args.output_folder_base, args.CHECKPOINTS_PREFIX, "%s_diff_adv" %
                                    self.ae_motion_type.value)
        else:
            dir_name = os.path.join(args.output_folder_base, args.CHECKPOINTS_PREFIX, "%s_diff_target" %
                                    self.ae_motion_type.value)
        create_dir(dir_name)
        log_message("Number of images: %d" % data_reader_.num_images)
        iterations = int(math.ceil(float(data_reader_.num_images) / batch_size))
        for iteration in range(0, iterations):
            batch_input, batch_target, file_paths = data_reader_.get_next_batch(iteration, batch_size,
                                                                                return_file_names=True)
            decoded_ = self.get_reconstructed_flow(batch_input)
            for idx, file_path in enumerate(file_paths):
                split_name = file_path.split(os.sep)
                
                name = split_name[-3] + "_" + split_name[-1][:-4] + '.npy'
                dst_file = os.path.join(dir_name, name)
                
                diff = np.abs(batch_target[idx] - (decoded_[idx]))
                np.save(dst_file, diff.flatten())

        log_function_end()

    def compute_latent_features_for_ae(self, data_reader_, is_adv=False, epoch=None, batch_size=64):
        log_function_start()
        self.restore_model(epoch)
        if is_adv:
            dir_name = os.path.join(args.output_folder_base, args.CHECKPOINTS_PREFIX, "%s_latent_adv" %
                                    self.ae_motion_type.value)
        else:
            dir_name = os.path.join(args.output_folder_base, args.CHECKPOINTS_PREFIX, "%s_latent_target" %
                                    self.ae_motion_type.value)
        create_dir(dir_name)
        log_message("Number of images: %d" % data_reader_.num_images)
        iterations = int(math.ceil(float(data_reader_.num_images) / batch_size))
        for iteration in range(0, iterations):
            batch_input, batch_target, file_paths = data_reader_.get_next_batch(iteration, batch_size,
                                                                                return_file_names=True)
            encoded_ = self.get_latent_features_feed_dict(batch_input)
            for idx, file_path in enumerate(file_paths):
                split_name = file_path.split(os.sep)

                name = split_name[-3] + "_" + split_name[-1][:-4] + '.npy'
                dst_file = os.path.join(dir_name, name)
                np.save(dst_file, encoded_[idx].flatten())

        log_function_end()


def get_tf_data_set(fn, batch_size):
    output_shape = [MotionAe.input_size[0], MotionAe.input_size[1], 2]

    tf_data_set = tf.data.Dataset.from_generator(fn,
                                                 output_types=(tf.float32, tf.float32, tf.string),
                                                 output_shapes=(tf.TensorShape(output_shape),
                                                                tf.TensorShape(output_shape), tf.TensorShape(None))) \
        .repeat().batch(batch_size)

    return tf_data_set


def train(ae_motion_type: MotionAeType, batch_size=64):
    log_function_start()
    tf.reset_default_graph()
    if ae_motion_type.value == MotionAeType.NEXT.value:
        folder_name = "optical_flow_samples_fwd"
    else:
        folder_name = "optical_flow_samples_bwd"

    data_reader_training: DataSetReaderMotionOpticalFlow = DataSetReaderMotionOpticalFlow(ae_motion_type,
                                                                                          os.path.join(
                                                                                              args.output_folder_base,
                                                                                              args.database_name,
                                                                                              ProcessingType.TRAIN.value),
                                                                                          folder_name,
                                                                                          input_size=MotionAe.input_size,
                                                                                          min_bbox_size=min_bbox_size,
                                                                                          max_bbox_size=max_bbox_size,
                                                                                          is_testing=False)

    if ae_motion_type.value == MotionAeType.NEXT.value:
        folder_name_adv = "optical_flow_samples_fwd_adv"
    else:
        folder_name_adv = "optical_flow_samples_bwd_adv"

    data_reader_training_adv: DataSetReaderAdversarialMotionOpticalFlow = DataSetReaderAdversarialMotionOpticalFlow(os.path.join(
                                                                                              args.output_folder_base,
                                                                                              args.database_name,
                                                                                              ProcessingType.TRAIN.value),
                                                                                          folder_name_adv)

    log_message("Number of train images: %d" % data_reader_training.num_images)
    ae_trainer = MotionAe(ae_motion_type)
    DataSetReaderMotionOpticalFlow.ADD_NOISE = False

    ae_trainer.train(data_reader_training, data_reader_training_adv)
    ae_trainer.compute_latent_features_for_ae(data_reader_training, False)
    ae_trainer.compute_latent_features_for_ae(data_reader_training_adv, True)

    ae_trainer.compute_reconstruction_features_for_ae(data_reader_training, False)
    ae_trainer.compute_reconstruction_features_for_ae(data_reader_training_adv, True)
    # ae_trainer.compute_max_error(data_reader_training)
    #
    # ae_trainer.compute_reconstruction_features(data_reader_training, batch_size=batch_size)
    # ae_trainer.compute_reconstruction_features(data_reader_training, batch_size=batch_size)
    # ae_trainer.visualise_reconstructed_images(data_reader_training, write_to_disk=True)
    ae_trainer.close_session()
    log_function_end()


def test(ae_motion_type: MotionAeType, batch_size=64):
    log_function_start()
    tf.reset_default_graph()
    if ae_motion_type.value == MotionAeType.NEXT.value:
        folder_name = "optical_flow_samples_fwd"
    else:
        folder_name = "optical_flow_samples_bwd"

    data_reader_test: DataSetReaderMotionOpticalFlow = DataSetReaderMotionOpticalFlow(ae_motion_type,
                                                                                      os.path.join(
                                                                                          args.output_folder_base,
                                                                                          args.database_name,
                                                                                          ProcessingType.TEST.value),
                                                                                      folder_name,
                                                                                      input_size=MotionAe.input_size,
                                                                                      min_bbox_size=min_bbox_size,
                                                                                      max_bbox_size=max_bbox_size,
                                                                                      is_testing=True)

    ae_trainer = MotionAe(ae_motion_type)
    ae_trainer.compute_latent_features_feed_dict(data_reader_test, batch_size=batch_size)
    ae_trainer.compute_reconstruction_features(data_reader_test, batch_size=batch_size)
    # ae_trainer.visualise_reconstructed_images(data_reader_test, write_to_disk=True)
    ae_trainer.close_session()
    log_function_end()



