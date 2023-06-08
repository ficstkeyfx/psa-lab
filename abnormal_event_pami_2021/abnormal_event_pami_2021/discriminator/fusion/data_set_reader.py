#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.


from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np
import os
import args
import pdb
import cv2 as cv

import utils


class DataSetReader:
    def __init__(self, latent_paths, recon_paths, labels, num_channels):
        self.latent_paths = np.array(latent_paths)
        self.recon_paths = np.array(recon_paths)
        self.labels = np.expand_dims(np.array(labels), axis=1)
        self.num_samples = len(labels)
        self.end_index = 0
        self.num_channels_latent = 16
        self.num_channels = num_channels

    def next_batch(self, bach_size=64):
        if self.end_index == self.num_samples:
            self.end_index = 0
            self.latent_paths, self.recon_paths, self.labels = shuffle(self.latent_paths, self.recon_paths, self.labels)

        start_index = self.end_index
        self.end_index += bach_size
        self.end_index = min(self.end_index, self.num_samples)

        latent = self.read_samples(self.latent_paths[start_index:self.end_index],
                                   reshape_param=(8, 8, self.num_channels_latent))
        recon = self.read_samples(self.recon_paths[start_index:self.end_index],
                                  reshape_param=(64, 64, self.num_channels))
        return latent, recon, self.labels[start_index:self.end_index]


    def read_samples(self, paths, reshape_param):
        samples = []
        for path_ in paths:
        
            sample = np.load(path_)        
            sample = sample.reshape(reshape_param)
            samples.append(sample)

        return np.array(samples)


def create_readers_split(prefix_ae, num_channels=2):
    train_images_recon_paths = []
    train_images_latent_paths = []

    labels = []
    target_folder_latent = os.path.join(args.output_folder_base, args.CHECKPOINTS_PREFIX,
                                        "%s_latent_target" % prefix_ae)
    target_folder_recon = os.path.join(args.output_folder_base, args.CHECKPOINTS_PREFIX,
                                       "%s_diff_target" % prefix_ae)
    files = os.listdir(target_folder_latent)

    for file in files:
        try:
            latent_path = os.path.join(target_folder_latent, file)
            recon_path = os.path.join(target_folder_recon, file)
            if os.path.exists(latent_path) and os.path.exists(recon_path):
                train_images_latent_paths.append(latent_path)
                train_images_recon_paths.append(recon_path)
                labels.append(0)
        except Exception as ex:
            utils.log_message(file)

    adv_folder_latent = os.path.join(args.output_folder_base, args.CHECKPOINTS_PREFIX,
                                     "%s_latent_adv" % prefix_ae)
    adv_folder_recon = os.path.join(args.output_folder_base, args.CHECKPOINTS_PREFIX,
                                    "%s_diff_adv" % prefix_ae)

    files = os.listdir(adv_folder_latent)
    for file in files:
        try:
            latent_path = os.path.join(adv_folder_latent, file)
            recon_path = os.path.join(adv_folder_recon, file)
            if os.path.exists(latent_path) and os.path.exists(recon_path):  
                train_images_latent_paths.append(latent_path)
                train_images_recon_paths.append(recon_path)
                labels.append(1)
        except Exception:
            utils.log_message(file)


    X_train_latent, X_val_latent, X_train_recon, X_val_recon, y_train, y_val = train_test_split(train_images_latent_paths,
                                                                                                train_images_recon_paths,
                                                                                                labels, test_size=0.1,
                                                                                                random_state=357)
    utils.log_message("labels training distribution" + str(np.bincount(y_train)))
    utils.log_message("labels validation distribution" + str(np.bincount(y_val)))
    reader_train = DataSetReader(X_train_latent, X_train_recon, y_train, num_channels=num_channels)
    reader_val = DataSetReader(X_val_latent, X_val_recon, y_val, num_channels=num_channels)

    return reader_train, reader_val

