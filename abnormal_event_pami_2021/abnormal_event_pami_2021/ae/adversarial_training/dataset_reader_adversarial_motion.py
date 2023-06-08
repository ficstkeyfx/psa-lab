#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import os
import glob
from sklearn.utils import shuffle
import cv2 as cv
import numpy as np
import pdb


class DataSetReaderAdversarialMotionOpticalFlow:
    def __init__(self, directory_base_name, folder_name,  input_size=(64, 64)):
        """
    
        :param directory_base_name: ...\output\avenue\train
        :param folder_name: images_3_3_0.50
        :param input_size: (64, 64)
        :param min_bbox_size: 
        :param max_bbox_size: 
        """
        self.directory_base_name = directory_base_name
        self.input_size = input_size
        self.images_paths = self.get_images_paths(directory_base_name, folder_name)
        self.index_train = 0
        self.num_images = len(self.images_paths)
        self.max_std = 0.15

    def get_images_paths(self, directory_base_name, folder_name):
        # "01"
        images_paths = []
        videos_list = os.listdir(directory_base_name)
        videos_list.sort()
        for video_name in videos_list:
            video_samples_full_path = os.path.join(directory_base_name, video_name)
            if os.path.isdir(video_samples_full_path):
                samples_full_path = glob.glob(os.path.join(directory_base_name, video_name, folder_name, '*.npy'))
                samples_full_path.sort()
                for sample_path in samples_full_path:
                    # if os.path.exists(sample_path[:-7] + "_mask.png") or self.is_testing:
                    images_paths.append(sample_path)

        # print(images_paths)
        return images_paths

    def get_next_batch(self, iteration, batch_size=32, return_file_names=False, add_noise=False):
        if iteration == 0:
            self.index_train = 0
            self.images_paths = shuffle(self.images_paths)

        end = self.index_train + batch_size
        if end > self.num_images:
            end = self.num_images
            batch_size = end - self.index_train

        input_images = np.zeros((batch_size, self.input_size[0], self.input_size[1], 2), np.float32)
        output_images = np.zeros((batch_size, self.input_size[0], self.input_size[1], 2), np.float32)
        start = self.index_train
        for idx in range(start, end):
            img = np.load(self.images_paths[idx])
            img = cv.resize(img, self.input_size, interpolation=cv.INTER_LINEAR)
            if add_noise:
                std = np.random.random() * self.max_std
                noise = np.random.normal(0, std, img.shape)
            else:
                noisy_input = img.copy()

            input_images[idx - start, :, :, :] = noisy_input
            output_images[idx - start, :, :, :] = img

        self.index_train = end

        if end == self.num_images:
            self.index_train = 0
            self.images_paths = shuffle(self.images_paths)

        if return_file_names:
            return input_images, output_images, self.images_paths[start: end]
        return input_images, output_images
