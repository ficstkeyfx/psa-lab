#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.


import os
import glob
from sklearn.utils import shuffle
import cv2 as cv
import numpy as np
import pdb


class DataSetReaderAdversarial:
    def __init__(self, directory_base_name, input_size=(64, 64)):
        """
    
        :param directory_base_name: ...\output\avenue\train
        :param folder_name: images_3_3_0.50
        :param input_size: (64, 64)
        :param min_bbox_size: 
        :param max_bbox_size: 
        """
        self.directory_base_name = directory_base_name
        self.input_size = input_size
        self.images_paths = os.listdir(directory_base_name)
        self.index_train = 0
        self.num_images = len(self.images_paths)
        self.max_std = 0.15

    def get_next_batch(self, iteration, batch_size=32, return_file_names=False, add_noise=False):
        if iteration == 0:
            self.index_train = 0
            self.images_paths = shuffle(self.images_paths)

        end = self.index_train + batch_size
        if end > self.num_images:
            end = self.num_images
            batch_size = end - self.index_train

        input_images = np.zeros((batch_size, self.input_size[0], self.input_size[1], 1), np.float32)
        output_images = np.zeros((batch_size, self.input_size[0], self.input_size[1], 1), np.float32)
        start = self.index_train
        for idx in range(start, end):
            img = cv.imread(os.path.join(self.directory_base_name, self.images_paths[idx]), cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, self.input_size, interpolation=cv.INTER_CUBIC) / 255.0
            img = np.clip(img, 0, 1)
            if add_noise:
                std = np.random.random() * self.max_std
                noise = np.random.normal(0, std, img.shape)
                noisy_input = np.clip(img + noise, 0, 1)
            else:
                noisy_input = img.copy()

            input_images[idx - start, :, :, 0] = noisy_input
            output_images[idx - start, :, :, 0] = img

        self.index_train = end

        if end == self.num_images:
            self.index_train = 0
            self.images_paths = shuffle(self.images_paths)

        if return_file_names:
            return input_images, output_images, self.images_paths[start: end]
        return input_images, output_images
