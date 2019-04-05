import os
import sys
import time
# from naoqi import ALProxy
import argparse

import numpy as np
import argparse
import re
import tensorflow as tf
import tensorflow.contrib.slim as slim
import scipy.misc
import matplotlib.pyplot as plt

sys.path.append('monodepth')
from monodepth.monodepth_model import *
from monodepth.monodepth_dataloader import *
from monodepth.average_gradients import *

parser = argparse.ArgumentParser(description='NAO Monodepth')

parser.add_argument('--ip', type=str, help='ip address of the robot', required=True)
parser.add_argument('--port', type=str, help='port of the robot', required=True)

args = parser.parse_args()

# Saves a single frame from the robot (or mocks it)
def saveFrame (path, mock = False):
    if mock == True:
        return
    
    # try:
    #     photoCaptureProxy = ALProxy("ALPhotoCapture", args.ip, args.port)
    #     photoCaptureProxy.setHalfPressEnabled(True)
    #     photoCaptureProxy.setPictureFormat("jpg")
    #     photoCaptureProxy.takePicture(path, "naomono", True) 
    # except Exception:
    #     print("Error when creating ALPhotoCapture proxy:")
    #     print(str(e))
    #     exit(1)

def post_process_disparity(disp):
    _, h, w = disp.shape
    l_disp = disp[0,:,:]
    r_disp = np.fliplr(disp[1,:,:])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (1.0 - l_mask - r_mask) * m_disp

def test_simple(params, inputParams):
    """Test function."""

    left  = tf.placeholder(tf.float32, [2, inputParams['input_height'], inputParams['input_width'], 3])
    model = MonodepthModel(params, "test", left, None)

    input_image = scipy.misc.imread(inputParams['image_path'], mode="RGB")
    original_height, original_width, num_channels = input_image.shape
    input_image = scipy.misc.imresize(input_image, [inputParams['input_height'], inputParams['input_width']], interp='lanczos')
    input_image = input_image.astype(np.float32) / 255
    input_images = np.stack((input_image, np.fliplr(input_image)), 0)

    # SESSION
    config = tf.ConfigProto(allow_soft_placement=True)
    sess = tf.Session(config=config)

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coordinator)

    # RESTORE
    restore_path = inputParams.checkpoint_path.split(".")[0]
    train_saver.restore(sess, restore_path)

    disp = sess.run(model.disp_left_est[0], feed_dict={left: input_images})
    disp_pp = post_process_disparity(disp.squeeze()).astype(np.float32)

    output_directory = os.path.dirname(inputParams['image_path'])
    output_name = os.path.splitext(os.path.basename(inputParams['image_path']))[0]

    np.save(os.path.join(output_directory, "{}_disp.npy".format(output_name)), disp_pp)
    disp_to_img = scipy.misc.imresize(disp_pp.squeeze(), [original_height, original_width])
    plt.imsave(os.path.join(output_directory, "{}_disp.png".format(output_name)), disp_to_img, cmap='plasma')

    print('done!')






# Capture an image from the robot
saveFrame("/tmp/", True)

# Transform the image

# TODO

params = monodepth_parameters(
    encoder='vgg',
    height=480,
    width=640,
    batch_size=2,
    num_threads=1,
    num_epochs=1,
    do_stereo=False,
    wrap_mode='border',
    use_deconv=False,
    alpha_image_loss=0,
    disp_gradient_loss_weight=0,
    lr_loss_weight=0,
    full_summary=False)

inputParams = {
    'image_path': '/tmp/naomono.jpg',
    'input_width': 640,
    'input_height': 480,
    'checkpoint_path': 'monodepth/models/model_kitti'
}

test_simple(params, inputParams)

tf.app.run()