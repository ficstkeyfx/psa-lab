#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.


import os
import glob
from sklearn.utils import shuffle
import cv2 as cv
import numpy as np
import pdb
from enum import Enum
import pdb

from utils import check_file_existence, log_message, get_file_name


class MotionAeType(Enum):
    PREVIOUS = "previous"
    NEXT = "next"


class DataSetReaderMotion:

    ADD_NOISE = False

    def __init__(self, ae_type:MotionAeType, directory_base_name, folder_name, input_size=(64, 64),
                 min_bbox_size=0, max_bbox_size=300):
        """
    
        :param directory_base_name: ...\output\avenue\train
        :param folder_name: images_3_3_0.50
        :param input_size: (64, 64)
        :param min_bbox_size: 
        :param max_bbox_size: 
        """
        self.ae_type = ae_type
        self.directory_base_name = directory_base_name
        self.input_size = input_size
        self.min_bbox_size = min_bbox_size
        self.max_bbox_size = max_bbox_size
        self.images_paths = self.get_images_paths(directory_base_name, folder_name)
        self.index_train = 0
        self.num_images = len(self.images_paths)
        self.max_std = 0.05

    def get_images_paths(self, directory_base_name, folder_name):
        # "01"
        images_paths = []
        videos_list = os.listdir(directory_base_name)
        videos_list.sort()
        for video_name in videos_list:
            video_samples_full_path = os.path.join(directory_base_name, video_name)
            if os.path.isdir(video_samples_full_path):
                samples_full_path = glob.glob(os.path.join(directory_base_name, video_name, folder_name, '*_01.png'))
                samples_full_path.sort()
                for sample_path in samples_full_path:
                    # check that the images 00 and 02 exist
                    prev_path = sample_path.replace("_01.png", "_00.png")
                    next_path = sample_path.replace("_01.png", "_02.png")

                    if check_file_existence(prev_path) is False or check_file_existence(next_path) is False:
                        log_message("Prev file or next file from %s does not exist." % sample_path)
                        continue

                    img = cv.imread(sample_path, cv.IMREAD_GRAYSCALE)
                    height = img.shape[0]
                    width = img.shape[1]
                    if (min(height, width) >= self.min_bbox_size) and (max(height, width) <= self.max_bbox_size):
                        images_paths.append(sample_path)

        return images_paths

    def generator(self):
        self.images_paths = shuffle(self.images_paths)

        for idx in range(0, self.num_images):
            current_img = cv.imread(self.images_paths[idx], cv.IMREAD_GRAYSCALE)
            current_img = cv.resize(current_img, self.input_size, interpolation=cv.INTER_CUBIC) / 255.0
            if self.ae_type == MotionAeType.NEXT:
                next_path = self.images_paths[idx].replace("_01.png", "_02.png")
                ref_img = cv.imread(next_path, cv.IMREAD_GRAYSCALE)
                ref_img = cv.resize(ref_img, self.input_size, interpolation=cv.INTER_CUBIC) / 255.0
            else:
                prev_path = self.images_paths[idx].replace("_01.png", "_00.png")
                ref_img = cv.imread(prev_path, cv.IMREAD_GRAYSCALE)
                ref_img = cv.resize(ref_img, self.input_size, interpolation=cv.INTER_CUBIC) / 255.0

            img = cv.absdiff(current_img, ref_img)
            if self.ADD_NOISE:
                std = np.random.random() * self.max_std
                noise = np.random.normal(0, std, img.shape)
                noisy_input = np.clip(img + noise, 0, 1)
            else:
                noisy_input = img.copy()

            img = np.expand_dims(img, axis=2)
            noisy_input = np.expand_dims(noisy_input, axis=2)
            yield noisy_input, img, self.images_paths[idx]

    def get_next_batch(self, iteration, batch_size=32, return_file_names=False):
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
            current_img = cv.imread(self.images_paths[idx], cv.IMREAD_GRAYSCALE)
            current_img = cv.resize(current_img, self.input_size, interpolation = cv.INTER_CUBIC) / 255.0
            if self.ae_type == MotionAeType.NEXT:
                next_path = self.images_paths[idx].replace("_01.png", "_02.png")
                ref_img = cv.imread(next_path, cv.IMREAD_GRAYSCALE)
                ref_img = cv.resize(ref_img, self.input_size, interpolation = cv.INTER_CUBIC) / 255.0
            else:
                prev_path = self.images_paths[idx].replace("_01.png", "_00.png")
                ref_img = cv.imread(prev_path, cv.IMREAD_GRAYSCALE)
                ref_img = cv.resize(ref_img, self.input_size, interpolation = cv.INTER_CUBIC) / 255.0

            img = cv.absdiff(current_img, ref_img)

            if self.ADD_NOISE:
                std = np.random.random() * self.max_std
                noise = np.random.normal(0, std, img.shape)
                noisy_input = np.clip(img + noise, 0, 1)
            else:
                noisy_input = img.copy()

            input_images[idx - start, :, :, 0] = noisy_input
            output_images[idx - start, :, :, 0] = img.copy()

        self.index_train = end
        if return_file_names:
            return input_images, output_images, self.images_paths[start: end]

        return input_images, output_images
