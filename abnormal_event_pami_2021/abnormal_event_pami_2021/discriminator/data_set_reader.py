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
    def __init__(self, train_images, labels):
        self.dataset = np.array(train_images)
        self.labels = np.expand_dims(np.array(labels), axis=1)
        self.num_samples = len(labels)
        self.end_index = 0

    def next_batch(self, bach_size=64):
        if self.end_index == self.num_samples:
            self.end_index = 0
            self.dataset, self.labels = shuffle(self.dataset, self.labels)

        start_index = self.end_index
        self.end_index += bach_size
        self.end_index = min(self.end_index, self.num_samples)

        return self.dataset[start_index:self.end_index], self.labels[start_index:self.end_index]


def create_readers(prefix_ae, num_channels=2):
    train_images = []
    labels = []
    target_folder = os.path.join(args.output_folder_base, args.CHECKPOINTS_PREFIX, "%s_latent_target" % prefix_ae)
    files = os.listdir(target_folder)
    # if prefix_ae.find('app') != -1:
    #     files = shuffle(files)
    #     files = files[:100000]

    for idx, file in enumerate(files):
        print(idx)
        sample = np.load(os.path.join(target_folder, file))
        train_images.append(sample.reshape(8, 8, num_channels))
        labels.append(0)

    adv_folder = os.path.join(args.output_folder_base, args.CHECKPOINTS_PREFIX, "%s_latent_adv" % prefix_ae)
    files = os.listdir(adv_folder)
    for file in files:
        print(file)
        sample = np.load(os.path.join(adv_folder, file))
        train_images.append(sample.reshape(8, 8, num_channels))
        labels.append(1)

    X_train, X_val, y_train, y_val = train_test_split(train_images, labels, test_size=0.1, random_state=357)
    utils.log_message("labels training distribution" + str(np.bincount(y_train)))
    utils.log_message("labels validation distribution" + str(np.bincount(y_val)))
    reader_train = DataSetReader(X_train, y_train)
    reader_val = DataSetReader(X_val, y_val)

    return reader_train, reader_val
 
