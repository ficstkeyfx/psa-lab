#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.


import args
import numpy as np
import os
import pdb
import tensorflow as tf
import cv2 as cv
from utils import crop_bbox, create_dir


def flow_to_color(flow, mask=None, max_flow=None):
    """Converts flow to 3-channel color image.

    Args:
        flow: tensor of shape [num_batch, height, width, 2].
        mask: flow validity mask of shape [num_batch, height, width, 1].
    """
    n = 8
    num_batch, height, width, _ = tf.unstack(tf.shape(flow))
    mask = tf.ones([num_batch, height, width, 1]) if mask is None else mask
    flow_u, flow_v = tf.unstack(flow, axis=3)
    if max_flow is not None:
        max_flow = tf.maximum(tf.to_float(max_flow), 1.)
    else:
        max_flow = tf.reduce_max(tf.abs(flow * mask))
    mag = tf.sqrt(tf.reduce_sum(tf.square(flow), 3))
    angle = tf.atan2(flow_v, flow_u)

    im_h = tf.mod(angle / (2 * np.pi) + 1.0, 1.0)
    im_s = tf.clip_by_value(mag * n / max_flow, 0, 1)
    im_v = tf.clip_by_value(n - im_s, 0, 1)
    im_hsv = tf.stack([im_h, im_s, im_v], 3)
    im = tf.image.hsv_to_rgb(im_hsv)
    return im * mask


def delete_files(dir_name):
    file_names = os.listdir(dir_name)
    for file_name in file_names:
        os.remove(os.path.join(dir_name, file_name))


def get_mag_rot(flow):
    mag, rot = cv.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    return np.dstack((mag, rot))


folder_name = 'train'
is_adv = False

if folder_name == 'test':
    video_dir = os.path.join(args.input_folder_base, folder_name, "frames")
    # video_names = [f[:-4] for f in os.listdir(video_dir)]
    video_names = os.listdir(video_dir)
else:
    video_dir = os.path.join(args.input_folder_base, folder_name, "videos")
    video_names = [f[:-4] for f in os.listdir(video_dir)]

meta_base_dir = os.path.join(args.output_folder_base, args.database_name, folder_name, "%s",
                             args.meta_folder_name, "%s")

video_names.sort()

for video_idx in range(len(video_names)):
    video_name = video_names[video_idx]
    print(video_name)
    if is_adv:
        optical_flow_base_dir = os.path.join(args.input_folder_base, folder_name, 'optical_flow_adv', video_name)
    else:
        optical_flow_base_dir = os.path.join(args.input_folder_base, folder_name, 'optical_flow', video_name)

    if video_idx + 1 != len(video_names):
        optical_flow_base_next_dir = optical_flow_base_dir.replace(video_name, video_names[video_idx + 1])
        if os.path.exists(optical_flow_base_next_dir) is False:
            break

    if len(os.listdir(optical_flow_base_dir)) == 0:
        continue

    print(optical_flow_base_dir)
    meta_files = os.listdir(meta_base_dir % (video_name, ""))
    frame_idx_motion = -1
    frame_motion_fwd = None
    frame_motion_bwd = None

    if is_adv:
        fwd_samples_base_dir = os.path.join(args.output_folder_base, args.database_name, folder_name, video_name,
                                            'optical_flow_samples_fwd_adv')
        bwd_samples_base_dir = os.path.join(args.output_folder_base, args.database_name, folder_name, video_name,
                                            'optical_flow_samples_bwd_adv')
    else:
        fwd_samples_base_dir = os.path.join(args.output_folder_base, args.database_name, folder_name, video_name,
                                            'optical_flow_samples_fwd')
        bwd_samples_base_dir = os.path.join(args.output_folder_base, args.database_name, folder_name, video_name,
                                            'optical_flow_samples_bwd')

    create_dir(fwd_samples_base_dir)
    create_dir(bwd_samples_base_dir)
    meta_files.sort()
    for meta_file in meta_files:
        meta = np.loadtxt(meta_base_dir % (video_name, meta_file))
        frame_idx = meta[0]
        bbox = meta[1:5]  # xmin ymin xmax ymax
        bbox = [int(b) for b in bbox]
        if frame_idx != frame_idx_motion:
            
            if os.path.exists(os.path.join(optical_flow_base_dir, '%d_of_fw.npy' % frame_idx)) is False:
                print('does not exist')
                continue
            frame_idx_motion = frame_idx
            frame_motion_fwd = np.load(os.path.join(optical_flow_base_dir, '%d_of_fw.npy' % frame_idx))
            frame_motion_bwd = np.load(os.path.join(optical_flow_base_dir, '%d_of_bw.npy' % frame_idx))

        crop_fwd = crop_bbox(frame_motion_fwd, bbox)
        crop_bwd = crop_bbox(frame_motion_bwd, bbox)
        mag_rot_fwd = get_mag_rot(crop_fwd)
        mag_rot_bwd = get_mag_rot(crop_bwd)
        # write them
        file_short_name = meta_file[:-4]
        
        np.save(os.path.join(fwd_samples_base_dir, file_short_name + ".npy"), mag_rot_fwd)
        np.save(os.path.join(bwd_samples_base_dir, file_short_name + ".npy"), mag_rot_bwd)

    delete_files(optical_flow_base_dir)
