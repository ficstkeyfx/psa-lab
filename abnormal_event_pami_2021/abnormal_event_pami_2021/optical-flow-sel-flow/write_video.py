import os
import numpy as np
import glob
import tensorflow as tf
import pdb
import cv2 as cv


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


# of_path = 'C:\\Research\\abnormal-event\\turing\\avenue\\optical-flow\\16_test_op\\optical_flow'
of_path = 'C:\\Research\\abnormal-event\\work-in-progress\\output\\avenue\\train\\01\\optical_flow'
direction = 'bw'

files_names = glob.glob(os.path.join(of_path, '*_%s.npy' % direction))

num_files = len(files_names)
flow_ph = tf.placeholder(tf.float32, shape=(1, None, None, 2), name="flow_ph")
img_color = flow_to_color(flow_ph, mask=None, max_flow=256)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
output_video = cv.VideoWriter('01_%s.mp4' % direction, 0x7634706d, 25, (640,  360))

for i in range(1, num_files + 1):
    flow = np.load(os.path.join(of_path, '%d_of_%s.npy' % (i, direction)))
    cv.cartToPolar(flow[:, :, 0], flow[:, :, 1])
    pdb.set_trace()
    img = sess.run(img_color, feed_dict={flow_ph: np.expand_dims(flow, axis=0)})
    img = np.uint8(img[0] * 255)
    img = img[:, :, [2, 1, 0]]
    output_video.write(img)

output_video.release()
