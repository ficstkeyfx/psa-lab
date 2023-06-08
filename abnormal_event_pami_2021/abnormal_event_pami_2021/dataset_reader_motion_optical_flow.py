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
import args

from utils import check_file_existence, log_message, get_file_name


class MotionAeType(Enum):
    PREVIOUS = "previous"
    NEXT = "next"


class DataSetReaderMotionOpticalFlow:

    ADD_NOISE = False

    def __init__(self, ae_type: MotionAeType, directory_base_name, folder_name, input_size=(64, 64),
                 min_bbox_size=30, max_bbox_size=300, is_testing=True):
        """
    
        :param directory_base_name: ...\output\avenue\train
        :param folder_name: optical_flow_samples_fwd or optical_flow_samples_bwd
        :param input_size: (64, 64)
        :param min_bbox_size: 
        :param max_bbox_size: 
        """
        self.is_testing = is_testing
        self.ae_type = ae_type
        self.directory_base_name = directory_base_name
        self.input_size = input_size
        self.min_bbox_size = min_bbox_size
        self.max_bbox_size = max_bbox_size
        self.folder_name = folder_name
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
                samples_full_path = glob.glob(os.path.join(directory_base_name, video_name, folder_name, '*.npy'))
                samples_full_path.sort()
                for sample_path in samples_full_path:
                    # if os.path.exists(sample_path[:-7] + "_mask.png") or self.is_testing:
                    if self.is_testing is False:
                        # pdb.set_trace()
                        meta_path = sample_path.replace(self.folder_name, args.meta_folder_name).replace(
                            '.npy', '.txt')
                        meta = np.loadtxt(meta_path)
                        class_id = meta[-2]
                        if class_id in args.excluded_training_classes:
                            # pdb.set_trace()
                            continue
                    images_paths.append(sample_path)

        # print(images_paths)
        return images_paths

    def generator(self):
        self.images_paths = shuffle(self.images_paths)
        for idx in range(0, self.num_images):
            
            img = np.load(self.images_paths[idx])
            img = cv.resize(img, self.input_size, interpolation=cv.INTER_LINEAR)
            if self.ADD_NOISE:
                std = np.random.random() * self.max_std
                noise = np.random.normal(0, std, img.shape)
                noisy_input = img + noise
            else:
                noisy_input = img.copy()

            yield noisy_input, img, self.images_paths[idx]

    def get_next_batch(self, iteration, batch_size=32, return_file_names=False):
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

            if self.ADD_NOISE:
                std = np.random.random() * self.max_std
                noise = np.random.normal(0, std, img.shape)
                noisy_input = img + noise
            else:
                noisy_input = img.copy()

            input_images[idx - start, :, :, :] = noisy_input
            output_images[idx - start, :, :, :] = img.copy()

        self.index_train = end
        if return_file_names:
            return input_images, output_images, self.images_paths[start: end]

        return input_images, output_images
