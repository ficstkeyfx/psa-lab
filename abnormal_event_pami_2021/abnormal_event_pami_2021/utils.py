#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.


import tensorflow as tf
from enum import Enum
import datetime
import os
import pdb
import sys
import numpy as np
from sklearn.svm import LinearSVC
import cv2 as cv

logs_folder = str
RUNNING_ID = str


def set_vars(logs_folder_, running_id):
    global logs_folder, RUNNING_ID
    logs_folder = logs_folder_
    RUNNING_ID = running_id


class ProcessingType(Enum):
    TRAIN = "train"
    TEST = "test"


def concat_images(pred, ground_truth):
    """
    :param input_image: imaginea grayscale (canalul L din reprezentarea Lab).
    :param pred: imaginea prezisa.
    :param ground_truth: imaginea ground-truth.
    :return: concatenarea imaginilor.
    """
    h, w, _ = pred.shape
    space_btw_images = int(0.2 * w)
    image = np.ones((h, w * 2 + 2 * space_btw_images, 3)) * 255
    # add ground truth
    image[:, :w] = ground_truth
    # add predicted
    offset = w + space_btw_images
    image[:, offset: offset + w] = pred
    return np.uint8(image)


def create_flow(image):
    mag = image[:, :, 0]
    angle = image[:, :, 1]
    max_flow = 16
    n = 8
    im_h = np.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = np.clip(mag * n / max_flow, 0, 1)
    im_v = np.clip(n - im_s, 0, 1)

    im_hsv = np.stack([im_h, im_s, im_v], 2)
    outimageHSV = np.uint8(im_hsv * 255)
    outimageBGR = cv.cvtColor(outimageHSV, cv.COLOR_HSV2BGR)
    return outimageBGR


class TemporalFrame:

    def __init__(self, temporal_size, max_size):
        self.temporal_size = temporal_size
        self.max_size = max_size
        self.frames = []

    def add(self, frame):
        self.frames.append(frame.copy())
        if len(self.frames) > self.max_size:
            self.frames.pop(0)

    def get(self, index):
        if index < 0:
            return self.frames[self.temporal_size + index].copy()
        if index >= 0:
            return self.frames[self.temporal_size + index].copy()

    def get_middle_frame(self):
        return self.frames[self.temporal_size].copy()


def crop_bbox(img, bbox):
    crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]].copy()
    return crop


def log_function_start():
    message = "Function %s has started." % sys._getframe().f_back.f_code.co_name
    file_handler = open(os.path.join(logs_folder, '%s_log.txt' % RUNNING_ID), 'a')
    file_handler.write("\n" + "=" * 30 + "\n\n" + "{} - {}".format(datetime.datetime.now(), message))
    file_handler.close()


def log_function_end():
    message = "Function %s has ended." % sys._getframe().f_back.f_code.co_name
    file_handler = open(os.path.join(logs_folder, '%s_log.txt' % RUNNING_ID), 'a')
    file_handler.write("\n" + "=" * 30 + "\n\n" + "{} - {}".format(datetime.datetime.now(), message))
    file_handler.close()


def log_message(message):
    print(message)
    file_handler = open(os.path.join(logs_folder, '%s_log.txt' % RUNNING_ID), 'a')
    file_handler.write("\n" + "=" * 30 + "\n\n" + "{} - {}".format(datetime.datetime.now(), message))
    file_handler.close()


def log_error(error):
    print('!!ERROR: ', error)
    file_handler = open('errors.txt', 'a')
    file_handler.write("\n" + "=" * 30 + "\n\n" + "{} - {}".format(datetime.datetime.now(), error))
    file_handler.close()


def load_graph(graph_path):
    with tf.gfile.GFile(graph_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    return graph_def


def read_graph_and_init_session(graph_path, name, config):
    graph_def = load_graph(graph_path)
    graph = tf.import_graph_def(graph_def, name=name)
    sess = tf.Session(graph=graph, config=config)
    return sess


def create_dir(directory_name):
    if not os.path.exists(directory_name):
        os.makedirs(directory_name)


def check_file_existence(file_path):
    return os.path.exists(file_path)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def train_linear_svm(x_train, labels, c):
    model = LinearSVC(penalty='l2', loss='squared_hinge', C=c, random_state=12)
    model.fit(x_train, labels)

    return model


def get_extension(file_name):
    if type(file_name) is str:
        return file_name.split('.')[-1]
    return None


def get_file_name(file_name):
    if type(file_name) is str:
        file_short_name, file_extension = os.path.splitext(file_name)
        return file_short_name
    return None
