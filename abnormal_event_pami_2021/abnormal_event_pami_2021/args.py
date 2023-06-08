#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.


import numpy as np
import tensorflow as tf
import os
from utils import ProcessingType, log_message, check_file_existence
import pdb
import sys
from utils import create_dir

operating_system = sys.platform

tf_config = tf.ConfigProto(device_count={'GPU': 1})
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.2
temporal_stride = 5
temporal_size = 5
assert temporal_size == temporal_stride
temporal_offsets = np.arange(-temporal_size, temporal_size + 1, temporal_stride)
detection_threshold = 0.8

lambda_ = 1.0

database_name = 'ShanghaiTech'
output_folder_base = '/home/lili/datasets/abnormal_event/shanghai/output_yolo_0.80'
input_folder_base = '/home/lili/datasets/abnormal_event/shanghai'
adversarial_images_path = "/home/lili/datasets/adversarial_images"

samples_folder_name = 'images_%d_%d_%.2f' % (temporal_size, temporal_stride, detection_threshold)
meta_folder_name = 'meta_%d_%d_%.2f' % (temporal_size, temporal_stride, detection_threshold)
object_detector_num_classes = 90
block_scale = 20
logs_folder = "logs"
num_samples_for_visualization = 500
CHECKPOINTS_PREFIX = 'adv_excluded_classes_unet_%f' % lambda_
excluded_training_classes = [2, 3, 4, 6, 8]
CHECKPOINTS_BASE = os.path.join(output_folder_base, database_name, "checkpoints", CHECKPOINTS_PREFIX)
create_dir(CHECKPOINTS_BASE)

allowed_video_extensions = ['avi', 'mp4']
allowed_image_extensions = ['jpg', 'png', 'jpeg']
RESTORE_FROM_HISTORY = True

history_filename = "history_%s_%s.txt" % (database_name, '%s')

if RESTORE_FROM_HISTORY is False:
    print('removing history...')
    if check_file_existence(history_filename % ProcessingType.TRAIN.value):
        os.remove(history_filename % ProcessingType.TRAIN.value)
    if check_file_existence(history_filename % ProcessingType.TEST.value):
        os.remove(history_filename % ProcessingType.TEST.value)


def log_parameters():
    message = "\n" * 5 + "Starting the algorithm with the following parameters: \n"
    local_vars = globals()
    for v in local_vars.keys():
        if not v.startswith('_'):
            message += " " * 5 + v + "=" + str(local_vars[v]) + "\n"
    log_message(message)

