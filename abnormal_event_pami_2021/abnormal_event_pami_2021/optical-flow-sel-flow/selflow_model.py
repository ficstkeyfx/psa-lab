# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
import os
import sys
import time
import timeit
import cv2

from six.moves import xrange
from scipy import misc, io
from tensorflow.contrib import slim

import args
from folder_images import *
# import matplotlib.pyplot as plt
from network import pyramid_processing
from datasets import BasicDataset
# from utils import average_gradients, lrelu, occlusion, rgb_bgr
import my_utils as my_utils
from data_augmentation import flow_resize
from flowlib import flow_to_color, write_flo
from warp import tf_warp
import pdb
# import imageio
from video import *


class SelFlowModel(object):
    def __init__(self, batch_size=8, iter_steps=1000000, initial_learning_rate=1e-4, decay_steps=2e5, 
                 decay_rate=0.5, is_scale=True, num_input_threads=4, buffer_size=5000,
                 beta1=0.9, num_gpus=1, save_checkpoint_interval=5000, write_summary_interval=200,
                 display_log_interval=50, allow_soft_placement=True, log_device_placement=False, 
                 regularizer_scale=1e-4, cpu_device='/cpu:0', save_dir='KITTI', checkpoint_dir='checkpoints', 
                 model_name='model', sample_dir='sample', summary_dir='summary', training_mode="no_distillation", 
                 is_restore_model=False, restore_model='./models/KITTI/no_census_no_occlusion',
                 dataset_config={}, self_supervision_config={}):
        self.batch_size = batch_size
        self.iter_steps = iter_steps
        self.initial_learning_rate = initial_learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.is_scale = is_scale
        self.num_input_threads = num_input_threads
        self.buffer_size = buffer_size
        self.beta1 = beta1       
        self.num_gpus = num_gpus
        self.save_checkpoint_interval = save_checkpoint_interval
        self.write_summary_interval = write_summary_interval
        self.display_log_interval = display_log_interval
        self.allow_soft_placement = allow_soft_placement
        self.log_device_placement = log_device_placement
        self.regularizer_scale = regularizer_scale
        self.training_mode = training_mode
        self.is_restore_model = is_restore_model
        self.restore_model = restore_model
        self.dataset_config = dataset_config
        self.self_supervision_config = self_supervision_config
        self.shared_device = '/gpu:0' if self.num_gpus == 1 else cpu_device
        assert(np.mod(batch_size, num_gpus) == 0)
        self.batch_size_per_gpu = int(batch_size / np.maximum(num_gpus, 1))
        
        self.save_dir = save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)         
        
        self.checkpoint_dir = '/'.join([self.save_dir, checkpoint_dir])
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir) 
        
        self.model_name = model_name
        if not os.path.exists('/'.join([self.checkpoint_dir, model_name])):
            os.makedirs(('/'.join([self.checkpoint_dir, self.model_name])))         
            
        self.sample_dir = '/'.join([self.save_dir, sample_dir])
        if not os.path.exists(self.sample_dir):
            os.makedirs(self.sample_dir)  
        if not os.path.exists('/'.join([self.sample_dir, self.model_name])):
            os.makedirs(('/'.join([self.sample_dir, self.model_name])))    
        
        self.summary_dir = '/'.join([self.save_dir, summary_dir])
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir) 
        if not os.path.exists('/'.join([self.summary_dir, 'train'])):
            os.makedirs(('/'.join([self.summary_dir, 'train']))) 
        if not os.path.exists('/'.join([self.summary_dir, 'test'])):
            os.makedirs(('/'.join([self.summary_dir, 'test'])))             
    
                    
    def test(self, restore_model, save_dir, is_normalize_img=True):
        dataset = BasicDataset(data_list_file=self.dataset_config['data_list_file'], img_dir=self.dataset_config['img_dir'], is_normalize_img=is_normalize_img)
        save_name_list = dataset.data_list[:, -1]
        iterator = dataset.create_one_shot_iterator(dataset.data_list, num_parallel_calls=self.num_input_threads)
        batch_img0, batch_img1, batch_img2 = iterator.get_next()
        img_shape = tf.shape(batch_img0)
        h = img_shape[1]
        w = img_shape[2]
        
        new_h = tf.where(tf.equal(tf.mod(h, 64), 0), h, (tf.to_int32(tf.floor(h / 64) + 1)) * 64)
        new_w = tf.where(tf.equal(tf.mod(w, 64), 0), w, (tf.to_int32(tf.floor(w / 64) + 1)) * 64)
        
        batch_img0 = tf.image.resize_images(batch_img0, [new_h, new_w], method=1, align_corners=True)
        batch_img1 = tf.image.resize_images(batch_img1, [new_h, new_w], method=1, align_corners=True)
        batch_img2 = tf.image.resize_images(batch_img2, [new_h, new_w], method=1, align_corners=True)
        
        flow_fw, flow_bw = pyramid_processing(batch_img0, batch_img1, batch_img2, train=False, trainable=False, is_scale=True) 
        flow_fw['full_res'] = flow_resize(flow_fw['full_res'], [h, w], method=1)
        flow_bw['full_res'] = flow_resize(flow_bw['full_res'], [h, w], method=1)
        
        flow_fw_color = flow_to_color(flow_fw['full_res'], mask=None, max_flow=256)
        flow_bw_color = flow_to_color(flow_bw['full_res'], mask=None, max_flow=256)
        
        restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
        saver = tf.train.Saver(var_list=restore_vars)
        sess = tf.Session()
        sess.run(tf.global_variables_initializer()) 
        sess.run(iterator.initializer) 
        saver.restore(sess, restore_model)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)           
        for i in range(dataset.data_num):
            start_time = timeit.default_timer()
            np_flow_fw, np_flow_bw, np_flow_fw_color, np_flow_bw_color = sess.run([flow_fw['full_res'], flow_bw['full_res'], flow_fw_color, flow_bw_color])
            end_time = timeit.default_timer()
            
            print('time ', end_time - start_time)
            
            pdb.set_trace()
            cv2.imwrite('%s/flow_fw_color_%s.png' % (save_dir, save_name_list[i]), np.uint8(np_flow_fw_color[0] * 255))
            cv2.imwrite('%s/flow_bw_color_%s.png' % (save_dir, save_name_list[i]), np.uint8(np_flow_bw_color[0] * 255))
            # misc.imsave('%s/flow_fw_color_%s.png' % (save_dir, save_name_list[i]), np_flow_fw_color[0])
            # misc.imsave('%s/flow_bw_color_%s.png' % (save_dir, save_name_list[i]), np_flow_bw_color[0])
            # write_flo('%s/flow_fw_%s.flo' % (save_dir, save_name_list[i]), np_flow_fw[0])
            # write_flo('%s/flow_bw_%s.flo' % (save_dir, save_name_list[i]), np_flow_bw[0])
            print('Finish %d/%d' % (i+1, dataset.data_num))    
         
    def mvn(self, img):
        # minus mean color and divided by standard variance
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        std = np.std(img, axis=(0, 1), keepdims=True) 
        img = (img - mean) / (std + 1e-12)
        return img

    def read_frames_from_video(self, video, temporal_frames, num_frames):
        for i in range(num_frames):
            if video.has_next:
                frame = video.read_frame()
                temporal_frames.add(frame)
            else:
                print("The video %s does not have enough frames." % video.name)

    def extract_flow(self, restore_model, save_dir, is_normalize_img=True):  
        batch_img0_h = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="batch_img0_h")
        batch_img1_h = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="batch_img1_h")
        batch_img2_h = tf.placeholder(tf.float32, shape=(None, None, None, 3), name="batch_img2_h")
        img_shape = tf.shape(batch_img0_h, name='shape')
        h = img_shape[1]
        w = img_shape[2]
        
        new_h = tf.where(tf.equal(tf.mod(h, 64), 0), h, (tf.to_int32(tf.floor(h / 64) + 1)) * 64)
        new_w = tf.where(tf.equal(tf.mod(w, 64), 0), w, (tf.to_int32(tf.floor(w / 64) + 1)) * 64)
        
        batch_img0 = tf.image.resize_images(batch_img0_h, [new_h, new_w], method=1, align_corners=True)
        batch_img1 = tf.image.resize_images(batch_img1_h, [new_h, new_w], method=1, align_corners=True)
        batch_img2 = tf.image.resize_images(batch_img2_h, [new_h, new_w], method=1, align_corners=True)
        
        flow_fw, flow_bw = pyramid_processing(batch_img0, batch_img1, batch_img2, train=False, trainable=False, is_scale=True) 
        flow_fw['full_res'] = flow_resize(flow_fw['full_res'], [h, w], method=1)
        flow_bw['full_res'] = flow_resize(flow_bw['full_res'], [h, w], method=1)
        
        flow_fw_color = flow_to_color(flow_fw['full_res'], mask=None, max_flow=256)
        flow_bw_color = flow_to_color(flow_bw['full_res'], mask=None, max_flow=256)
        
        restore_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) 
        saver = tf.train.Saver(var_list=restore_vars)
        tf_config = tf.ConfigProto(device_count={'GPU': 1})
        tf_config.gpu_options.per_process_gpu_memory_fraction = 0.5
        sess = tf.Session(config=tf_config)
        sess.run(tf.global_variables_initializer())  
        saver.restore(sess, restore_model)
        is_adv = False
        folder_name = 'train'
        if folder_name == 'test':
            assert is_adv is False

        if folder_name.find('train') != -1:
            video_dir = os.path.join(args.input_folder_base, folder_name, "videos")
            video_names = os.listdir(video_dir)
        else:
            video_dir = os.path.join(args.input_folder_base, folder_name, "frames")
            video_names = [f for f in os.listdir(video_dir) if os.path.isdir(os.path.join(video_dir, f))]

        video_names.sort()
    
        for video_idx, video_name in enumerate(video_names):
            print(video_idx)
            if folder_name.find('train') == -1:
                video = FolderImage(os.path.join(video_dir, video_name))
            else:
                video = Video(os.path.join(video_dir, video_name))

            if is_adv:
                images_output_dir = os.path.join(args.input_folder_base, folder_name, 'optical_flow_adv', video.name)
                temporal_frames = my_utils.TemporalFrame(temporal_size=6, max_size=13)
                frame_idx = 5
            else:
                images_output_dir = os.path.join(args.input_folder_base, folder_name, 'optical_flow', video.name)
                temporal_frames = my_utils.TemporalFrame(temporal_size=1, max_size=3)
                frame_idx = 0

            my_utils.create_dir(images_output_dir)

            self.read_frames_from_video(video, temporal_frames, temporal_frames.max_size - 1)   # fill the queue - 1

            while video.has_next:
                # read a frame and add to queue
                frame = video.read_frame()
                if frame is None:
                    break
                temporal_frames.add(frame)
                if is_adv:
                    stride = np.random.randint(4, 7)
                else:
                    stride = 1

                frame_idx += 1
                img_0 = temporal_frames.get(-stride)
                img_1 = temporal_frames.get_middle_frame()
                img_2 = temporal_frames.get(stride)

                img_0 = np.float32(img_0) / 255
                img_0 = self.mvn(img_0)
                img_1 = np.float32(img_1) / 255
                img_1 = self.mvn(img_1)
                img_2 = np.float32(img_2) / 255
                img_2 = self.mvn(img_2)

                img_0 = np.expand_dims(img_0, axis=0)
                img_1 = np.expand_dims(img_1, axis=0)
                img_2 = np.expand_dims(img_2, axis=0)

                feed_dict={batch_img0_h: img_0, batch_img1_h: img_1, batch_img2_h: img_2}
                np_flow_fw, np_flow_bw, np_flow_fw_color, np_flow_bw_color = sess.run([flow_fw['full_res'], flow_bw['full_res'], flow_fw_color, flow_bw_color], feed_dict=feed_dict)
                # res = np_flow_fw_color[0][:, :, [2, 1, 0]]
                # cv2.imshow('a', np.uint8(res * 255));
                # res2 = np_flow_bw_color[0][:, :, [2, 1, 0]]
                # cv2.imshow('b', np.uint8(res2 * 255));
                # cv2.waitKey(0)
                np.save(os.path.join(images_output_dir, '%d_of_fw.npy' % frame_idx), np_flow_fw[0])
                np.save(os.path.join(images_output_dir, '%d_of_bw.npy' % frame_idx), np_flow_bw[0])


        
    

        
            
              
