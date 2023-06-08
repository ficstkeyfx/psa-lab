import tensorflow as tf
from enum import Enum
import datetime
import os
import pdb
import sys
import numpy as np
from sklearn.svm import LinearSVC

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
    return
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
