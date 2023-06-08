#  2021, A Background-Agnostic Framework with Adversarial Training for Abnormal Event Detection in Video
#  Mariana-Iuliana Georgescu, Radu Tudor Ionescu, Fahad Shahbaz Khan, Marius Popescu and Mubarak Shah, TPAMI
#  SecurifAI’s NonCommercial Use & No Sharing International Public License.


import os

import cv2 as cv
import numpy as np

import args
from utils import crop_bbox

folder_name = 'train'
video_dir = os.path.join(args.input_folder_base, folder_name, "videos")
# video_names = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]
video_names = os.listdir(video_dir)

meta_base_dir = os.path.join(args.output_folder_base, args.database_name, folder_name, "%s",
                             args.meta_folder_name, "%s")

video_names.sort()
for video_name in video_names:
    video_name = video_name[:-4]
    print(video_name)
    masks_base_dir = os.path.join(args.input_folder_base, folder_name, 'masks', video_name,
                                  'masks_%.2f' % args.detection_threshold)
    meta_files = os.listdir(meta_base_dir % (video_name, ""))
    frame_idx_motion = -1
    frame_mask = None
    masks_output_dir = os.path.join(args.output_folder_base, args.database_name, folder_name, "%s",
                                    args.samples_folder_name) % video_name

    for meta_file in meta_files:
        meta = np.loadtxt(meta_base_dir % (video_name, meta_file))
        frame_idx = meta[0]
        bbox = meta[1:5]  # xmin ymin xmax ymax
        bbox = [int(b) for b in bbox]
        if frame_idx != frame_idx_motion:
            frame_idx_motion = frame_idx
            frame_mask = cv.imread(os.path.join(masks_base_dir, '%05d.png' % frame_idx), 0)

        # print(frame_idx)
    
        crop_mask = crop_bbox(frame_mask, bbox)
        if crop_mask.max() == 0:
            continue

        res = np.bincount(crop_mask.flatten())
        res = res.argsort()
        num = res[-1]
        if num == 0:
            num = res[-2]
        mask = (crop_mask == num) * 255

        # write them
        file_short_name = meta_file[:-4]
        cv.imwrite(os.path.join(masks_output_dir, file_short_name + "_mask.png"), mask)
        
        # only for visualization
        # img = sess.run(img_color, feed_dict={flow_ph: np.expand_dims(frame_motion_fwd, axis=0)})
        # img = np.uint8(img[0] * 255)
        # img = img[:, :, [2, 1, 0]]
        # bbox = [int(b) for b in bbox]
        # img = img.copy()
        # cv.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), thickness=2)
        # cv.imshow("img", img)
        # cv.waitKey(0)
