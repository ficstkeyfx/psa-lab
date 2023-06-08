#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import datetime
import sys

import restore_helper
from folder_images import *
from obj_det.mask_rcnn import *
from video import *

operating_system = sys.platform

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

if operating_system.find("win") == -1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def read_frames_from_video(video, temporal_frames, num_frames):
    for i in range(num_frames):
        if video.has_next:
            frame = video.read_frame()
            temporal_frames.add(frame)
        else:
            utils.log_error("The video %s does not have enough frames." % video.name)


def extract_masks(processing_type: utils.ProcessingType, is_video=True):
    """
    : It is the name of the folder
    :param is_video: if the frames are already extracted or not.
    :param processing_type: a/b/c/AVENUE (database_name)/TRAIN(folder_name)/videos/vid1
                                                                                   /vid2
                                                                                   /..
                                                                                   /vidN
    :return:
    """

    utils.log_function_start()
    folder_name = processing_type.value
    # read all the video names (video names)
    if is_video:
        video_dir = os.path.join(args.input_folder_base, folder_name, "videos")
        video_names = [f for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))
                       and utils.get_extension(f) in args.allowed_video_extensions]
    else:
        video_dir = os.path.join(args.input_folder_base, folder_name, "frames")
        video_names = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
    
    video_names.sort()
    num_videos = len(video_names)
    utils.log_message("Reading %d video names from %s." % (num_videos, video_dir))
    if args.RESTORE_FROM_HISTORY:
        video_names = restore_helper.restore_from_history(args.history_filename % processing_type.value, video_names)
    num_videos = len(video_names)
    utils.log_message("There are %d video in %s after reading the history file." % (num_videos, video_dir))

    obj_detector = MaskRCNN(args.detection_threshold, config=args.tf_config)
    for video_idx, video_name in enumerate(video_names):
        utils.log_message("Processing video %s, %d/%d.." % (video_name, video_idx, num_videos))
        if is_video:
            video = Video(os.path.join(video_dir, video_name))
        else:
            video = FolderImage(os.path.join(video_dir, video_name))

        if video.is_valid is False:
            restore_helper.add_to_history(args.history_filename % processing_type.value, video_name)
            continue

        # create output dir
        images_output_dir = os.path.join(args.input_folder_base, folder_name, 'masks', video.name,
                                         'masks_%.2f' % args.detection_threshold)
        utils.create_dir(images_output_dir)
        meta_output_dir = os.path.join(args.output_folder_base, args.database_name, folder_name, video.name,
                                       args.meta_folder_name)
        utils.create_dir(meta_output_dir)

        temporal_frames = utils.TemporalFrame(temporal_size=args.temporal_size, max_size=2 * args.temporal_size + 1)
        read_frames_from_video(video, temporal_frames, temporal_frames.max_size - 1)  # fill the queue - 1
        frame_idx = args.temporal_size - 1

        while video.has_next:
            # read a frame and add to queue
            frame = video.read_frame()
            if frame is None:
                break
            temporal_frames.add(frame)
            frame_idx += 1
            frame_to_process = temporal_frames.get_middle_frame()
            detections = obj_detector.get_detections(frame_to_process)
            blank_frame = np.zeros((frame_to_process.shape[0:2]))
            for idx_detection, detection in enumerate(detections):
                x_min, y_min, x_max, y_max = detection.get_bbox_as_list()
                # blank_frame[y_min: y_max, x_min: x_max] =
                bbox_mask = (cv.resize(detection.mask,  (detection.get_width(), detection.get_height()), interpolation=cv.INTER_NEAREST) / 255) * (idx_detection + 1)
                blank_frame[y_min: y_max, x_min: x_max] = np.maximum(blank_frame[y_min: y_max, x_min: x_max], bbox_mask)
            cv.imwrite(os.path.join(images_output_dir, '%05d.png' % frame_idx), np.uint8(blank_frame))

    utils.log_function_end()


# do not delete this
RUNNING_ID = str(datetime.datetime.now()).replace(" ", "_").replace(":", "-")
utils.set_vars(args.logs_folder, RUNNING_ID)
utils.create_dir(args.logs_folder)
args.log_parameters()
extract_masks(utils.ProcessingType.TRAIN, is_video=True)
