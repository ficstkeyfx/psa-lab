#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.


import os
import glob
from sklearn.utils import shuffle
import cv2 as cv
import numpy as np
from utils import log_message
import args
import pdb


class DataSetReaderAppearance:
    def __init__(self, directory_base_name, folder_name, input_size=(32, 32), min_bbox_size=30, max_bbox_size=300,
                 is_testing=True):
        """
    
        :param directory_base_name: ...\output\avenue\train
        :param folder_name: images_3_3_0.50
        :param input_size: (64, 64)
        :param min_bbox_size: 
        :param max_bbox_size: 
        """
        self.directory_base_name = directory_base_name
        self.input_size = input_size
        self.min_bbox_size = min_bbox_size
        self.max_bbox_size = max_bbox_size
        self.is_testing = is_testing
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
                samples_full_path = glob.glob(os.path.join(directory_base_name, video_name, folder_name, '*_01.png'))
                samples_full_path.sort()
                for sample_path in samples_full_path:
                    
                    if self.is_testing is False:
                        meta_path = sample_path.replace(args.samples_folder_name, args.meta_folder_name).replace('_01.png', '.txt')
                        
                        meta = np.loadtxt(meta_path)
                        class_id = meta[-2]
                        if class_id in args.excluded_training_classes:
                            # pdb.set_trace()
                            continue

                    img = cv.imread(sample_path, cv.IMREAD_GRAYSCALE)
                    height = img.shape[0]
                    width = img.shape[1]
                    if (max(height, width) <= self.max_bbox_size or self.is_testing) and (os.path.exists(sample_path[:-7] + "_mask.png") or self.is_testing):
                        images_paths.append(sample_path)
                        print(sample_path)
        log_message('num of images' + str(len(images_paths)))
        return images_paths

    def get_next_batch(self, iteration, batch_size=32, return_file_names=False, add_noise=False):
        if iteration == 0:
            self.index_train = 0
            self.images_paths = shuffle(self.images_paths)

        end = self.index_train + batch_size
        if end > self.num_images:
            end = self.num_images
            batch_size = end - self.index_train

        input_images = np.zeros((batch_size, self.input_size[0], self.input_size[1], 1), np.float32)
        mask_images = np.zeros((batch_size, self.input_size[0], self.input_size[1], 1), np.float32)
        output_images = np.zeros((batch_size, self.input_size[0], self.input_size[1], 1), np.float32)
        start = self.index_train
        for idx in range(start, end):
            img = cv.imread(self.images_paths[idx], cv.IMREAD_GRAYSCALE)
            img = cv.resize(img, self.input_size, interpolation=cv.INTER_CUBIC) / 255.0
            img = np.clip(img, 0, 1)
            mask = np.zeros(self.input_size)
            # TODO: don t forget to load the mask
            if self.is_testing is False:
                mask = cv.imread(self.images_paths[idx][:-7] + "_mask.png", cv.IMREAD_GRAYSCALE)
                mask = cv.resize(mask, self.input_size, interpolation=cv.INTER_LINEAR) / 255.0
                mask[mask < 0.5] = 0
                mask[mask >= 0.5] = 1

            if add_noise:
                std = np.random.random() * self.max_std
                noise = np.random.normal(0, std, img.shape)
                noisy_input = np.clip(img + noise, 0, 1)
            else:
                noisy_input = img.copy()

            input_images[idx - start, :, :, 0] = noisy_input
            mask_images[idx - start, :, :, 0] = mask
            output_images[idx - start, :, :, 0] = img

        self.index_train = end
        if return_file_names:
            return input_images, output_images, self.images_paths[start: end]
        return input_images, output_images, mask_images
