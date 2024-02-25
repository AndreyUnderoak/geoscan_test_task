# This module is a part of geoscan task
#
# Made by Andrey Underoak https://github.com/AndreyUnderoak

import cv2
from ToGray import ToGray
from ImgWorker import ImgWorker
import glob as gl
import numpy as np
import sys

# default pixels for model
model_px = 128

# if arg == 512 use 512 model
if(len(sys.argv) >1):
    if(int(sys.argv[1]) == 512):
        model_px = 512

# task goal output
goal_px = 2048
path = "./input_images/*.png"
train = False

filenames = gl.glob(path)

# TODO do train
if(train):
    pass

# init saved model
tg = ToGray("./pretrained_"+str(model_px)+"/ColorToGray.ckpt-49", model_px)

# load input
input_images = ImgWorker.load_img(filenames, model_px)
input_images_np = np.asarray(input_images)   

# use model to gray images
gray_images = tg.color_to_gray_array(input_images)

# smoothing images
smoothed_images = ImgWorker.smooth_resize(gray_images, goal_px)

# calculate gradients
x_grad = ImgWorker.to_grad_x(smoothed_images)
y_grad = ImgWorker.to_grad_y(smoothed_images)

# deleting zeros by norming
x_norm = ImgWorker.normalize(x_grad)
y_norm = ImgWorker.normalize(y_grad)

# save output
ImgWorker.save_img(x_norm, "x_norm", model_px)
ImgWorker.save_img(y_norm, "y_norm", model_px)


cv2.waitKey(0)
