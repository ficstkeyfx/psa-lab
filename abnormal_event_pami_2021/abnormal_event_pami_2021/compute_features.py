#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.

import os
import numpy as np
import sklearn.cluster as sc
from sklearn.metrics import roc_curve, auc
import timeit
import pickle
import pdb
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import convolve
import math
import cv2 as cv
import random 

from utils import log_function_start, log_function_end, log_message, create_dir, ProcessingType, train_linear_svm,\
    check_file_existence
import args


def compute_abnormality_scores(processing_type: ProcessingType, save_per_video=True):
    log_function_start() 
    sample_base_dir = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                                   args.samples_folder_name, "%s")
    sample_base_dir_fwd = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                                       "optical_flow_samples_fwd", "%s")
    sample_base_dir_bwd = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                                       "optical_flow_samples_bwd", "%s")
    meta_base_dir = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                                 args.meta_folder_name, "%s")

    appearance_base_dir = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                                       "appearance_latent_features_unet_%f" % args.lambda_, "%s")
    appearance_base_dir_recon = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                                             "appearance_reconstruction_features_unet_%f" % args.lambda_, "%s")

    motion_next_base_dir = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                                        "motion_latent_features_unet_%f_next" % args.lambda_, "%s")
    motion_next_base_dir_recon = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                                              "motion_reconstruction_features_unet_%f_next" % args.lambda_, "%s")

    motion_prev_base_dir = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                                        "motion_latent_features_unet_%f_previous" % args.lambda_, "%s")
    motion_prev_base_dir_recon = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                                              "motion_reconstruction_features_unet_%f_previous" % args.lambda_, "%s")

    # C:\Research\abnormal - event\work - in -progress\output\avenue\train
    videos_features_base_dir = os.path.join(args.output_folder_base, args.database_name, processing_type.value)
    videos_names = os.listdir(videos_features_base_dir)
    videos_names.sort()

    concat_features_path = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                                        "anormality_scores_%f.txt" % args.lambda_)
    loc_v3_path = os.path.join(args.output_folder_base, args.database_name, processing_type.value, "%s",
                               "loc_v3_%f.npy" % args.lambda_)

    import discriminator.fusion.trainer_discriminator as trainer_discriminator_fusion
    discriminator = trainer_discriminator_fusion.Experiment("app", is_testing=True, num_channels=1)
    discriminator_prev = trainer_discriminator_fusion.Experiment("previous", is_testing=True, num_channels=2)
    discriminator_next = trainer_discriminator_fusion.Experiment("next", is_testing=True, num_channels=2)
    all_features = []
    input_size = (64, 64)
    for video_name in videos_names:  # for each video
        # check if it is a dir
        if os.path.isdir(os.path.join(videos_features_base_dir, video_name)) is False:
            continue
        log_message(video_name)
        # read all the appearance features
        samples_names = os.listdir(meta_base_dir % (video_name, ""))  # maybe this is a hack :)
        samples_names.sort()
        if save_per_video:
            video_features_path = concat_features_path % video_name
            video_loc_v3_path = loc_v3_path % video_name

        features_video = []
        loc_v3_video = []
        for sample_name in samples_names:
            # read the images
            short_sample_name = sample_name[:sample_name.rfind(".")]
            
            meta = np.loadtxt(meta_base_dir % (video_name, sample_name.replace("_01", "")))

            img_1 = cv.imread(sample_base_dir % (video_name, short_sample_name + "_01.png"), cv.IMREAD_GRAYSCALE)

            img_1 = cv.resize(img_1, (64, 64), interpolation=cv.INTER_CUBIC) / 255

            img_prev = np.load(sample_base_dir_bwd % (video_name, short_sample_name + ".npy"))
            img_next = np.load(sample_base_dir_fwd % (video_name, short_sample_name + ".npy"))
            img_prev = cv.resize(img_prev, input_size, interpolation=cv.INTER_LINEAR)
            img_next = cv.resize(img_next, input_size, interpolation=cv.INTER_LINEAR)

            if not (check_file_existence(appearance_base_dir % (video_name, sample_name.replace('.txt', '.npy')))
                    and check_file_existence(motion_next_base_dir % (video_name, sample_name.replace('.txt', '.npy'))) and
                                             check_file_existence(motion_prev_base_dir % (video_name, sample_name.replace('.txt', '.npy')))):
                log_message(video_name + sample_name)
                continue
             
            err = 0
            appearance_features = np.load(appearance_base_dir % (video_name, sample_name.replace('.txt', '.npy')))
            appearance_features_recon = np.load(appearance_base_dir_recon % (video_name, sample_name.replace('.txt', '.npy')))
            appearance_features_recon = np.reshape(appearance_features_recon, input_size) / 255
            diff = np.abs(appearance_features_recon - img_1)
             
            err += (discriminator.predict(diff, appearance_features.reshape(8, 8, 16)))
            
            prev_motion_features = np.load(motion_prev_base_dir % (video_name, sample_name.replace('.txt', '.npy')))
            prev_motion_features_recon = np.load(motion_prev_base_dir_recon % (video_name, sample_name.replace('.txt', '.npy')))
            diff_prev = np.abs(img_prev - prev_motion_features_recon.reshape(64, 64, 2))

            err += (discriminator_prev.predict(diff_prev, prev_motion_features.reshape(8, 8, 16)))

            next_motion_features = np.load(motion_next_base_dir % (video_name, sample_name.replace('.txt', '.npy')))
            next_motion_features_recon = np.load(motion_next_base_dir_recon % (video_name, sample_name.replace('.txt', '.npy')))
            diff_next = np.abs(img_next - next_motion_features_recon.reshape(64, 64, 2))

            err += (discriminator_next.predict(diff_next, next_motion_features.reshape(8, 8, 16)))

            features = [err / 3]
            if save_per_video:
                features_video.append(features)
                loc_v3_video.append(meta[:-2])
            else:
                all_features.append(features)

        if save_per_video:
            np.savetxt(video_features_path, features_video)
            np.save(video_loc_v3_path, loc_v3_video)

    # save the features
    if not save_per_video:
        np.save(concat_features_path, all_features)
    log_function_end()


def gaussian_filter_3d(sigma=1.0):
    x = np.array([-2, -1, 0, 1, 2])
    f = np.exp(- (x ** 2) / (2 * (sigma ** 2))) / (sigma * np.sqrt(2 * np.pi))
    f += (1 - np.sum(f)) / len(f)
    k = np.expand_dims(f, axis=1).T * np.expand_dims(f, axis=1)
    k3d = np.expand_dims(k, axis=2).T * np.expand_dims(np.expand_dims(f, axis=1), axis=2)
    # k3d = k3d * 3
    return k3d


def gaussian_filter_(support, sigma):
    mu = support[len(support) // 2 - 1]
    filter = 1.0 / (sigma * np.sqrt(2 * math.pi)) * np.exp(-0.5 * ((support - mu) / sigma) ** 2)
    return filter


def predict_anomaly_on_frames(video_info_path, filter_3d, filter_2d):
    video_normality_scores = np.loadtxt(os.path.join(video_info_path, "anormality_scores_%f.txt" % args.lambda_))
    video_loc_v3 = np.load(os.path.join(video_info_path, "loc_v3_%f.npy" % args.lambda_))
    video_meta_data = pickle.load(open(os.path.join(video_info_path, "video_meta_data.pkl"), 'rb'))
    video_height = video_meta_data["height"]
    video_width = video_meta_data["width"]

    block_scale = args.block_scale
    block_h = int(round(video_height / block_scale))
    block_w = int(round(video_width / block_scale))

    anomaly_scores = video_normality_scores - min(video_normality_scores)
    anomaly_scores = anomaly_scores / max(anomaly_scores)

    num_frames = video_meta_data["num_frames"]
    num_bboxes = len(anomaly_scores)

    ab_event = np.zeros((block_h, block_w, num_frames))
    for i in range(num_bboxes):
        loc_V3 = np.int32(video_loc_v3[i])

        ab_event[int(round(loc_V3[2] / block_scale)): int(round(loc_V3[4] / block_scale) + 1),
        int(round(loc_V3[1] / block_scale)): int(round(loc_V3[3] / block_scale) + 1), loc_V3[0]] = np.maximum(
            ab_event[int(round(loc_V3[2] / block_scale)):int(round(loc_V3[4] / block_scale)) + 1,
            int(round(loc_V3[1] / block_scale)): int(round(loc_V3[3] / block_scale) + 1),
            loc_V3[0]], anomaly_scores[i])
    dim = 9
    filter_3d = np.ones((dim, dim, dim)) / (dim ** 3)
    ab_event3 = ab_event.copy()  # convolve(ab_event, filter_3d)  #
    np.save(os.path.join(video_info_path, 'ab_event3_%f.npy' % args.lambda_), ab_event3)
    frame_scores = np.zeros(num_frames)
    for i in range(num_frames):
        frame_scores[i] = ab_event3[:, :, i].max()

    padding_size = len(filter_2d) // 2
    in_ = np.concatenate((np.zeros(padding_size), frame_scores, np.zeros(padding_size)))
    frame_scores = np.correlate(in_, filter_2d, 'valid')
    return frame_scores


def compute_performance_indices(processing_type:ProcessingType=ProcessingType.TEST):
    log_function_start()
    filter_3d = gaussian_filter_3d(sigma=5)  # don't use it here
    filter_2d = gaussian_filter_(np.arange(1, 302), 21)
    # list all the testing videos
    videos_features_base_dir = os.path.join(args.output_folder_base, args.database_name, processing_type.value)
    testing_videos_names =[name for name in os.listdir(videos_features_base_dir) if os.path.isdir(os.path.join(videos_features_base_dir, name))]
    testing_videos_names.sort()
    all_frame_scores = []
    all_gt_frame_scores = []
    roc_auc_videos = []
    for video_name in testing_videos_names:
        log_message(video_name)
        video_scores = predict_anomaly_on_frames(os.path.join(videos_features_base_dir, video_name), filter_3d, filter_2d)
        all_frame_scores = np.append(all_frame_scores, video_scores)
        # read the ground truth scores at frame level
        gt_scores = np.loadtxt(os.path.join(videos_features_base_dir, video_name, "ground_truth_frame_level.txt"))
        all_gt_frame_scores = np.append(all_gt_frame_scores, gt_scores)
        fpr, tpr, _ = roc_curve(np.concatenate(([0], gt_scores, [0])), np.concatenate(([0], video_scores, [0])))
        roc_auc = auc(fpr, tpr)
        roc_auc_videos.append(roc_auc)

    fpr, tpr, _ = roc_curve(np.concatenate(([0], all_gt_frame_scores, [0])), np.concatenate(([0], all_frame_scores, [0])))
    roc_auc = auc(fpr, tpr)
    log_message("Frame-based AUC is %.3f on %s (all data set)." % (roc_auc, args.database_name))
    log_message("Avg. (on video) frame-based AUC is %.3f on %s." % (np.array(roc_auc_videos).mean(), args.database_name))
    log_function_end()

