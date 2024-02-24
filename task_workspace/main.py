import cv2
from ToGray import ToGray
from ImgWorker import ImgWorker
import glob as gl
import numpy as np
import sys

model_px = 128

if(len(sys.argv) >1):
    if(int(sys.argv[1]) == 512):
        model_px = 512

goal_px = 2048
path = "./input_images/*.png"
train = False

# count = 0
filenames = gl.glob(path)

if(train):
    pass

#Init saved model
tg = ToGray("./pretrained_"+str(model_px)+"/ColorToGray.ckpt-49", model_px)

input_images = ImgWorker.load_img(filenames, model_px)

input_images_np = np.asarray(input_images)   

gray_images = tg.color_to_gray_array(input_images)
smoothed_images = ImgWorker.smooth_resize(gray_images, goal_px)

x_grad = ImgWorker.to_grad_x(smoothed_images)
y_grad = ImgWorker.to_grad_y(smoothed_images)

x_norm = ImgWorker.normalize(x_grad)
y_norm = ImgWorker.normalize(y_grad)

ImgWorker.save_img(x_norm, "x_norm", model_px)
ImgWorker.save_img(y_norm, "y_norm", model_px)


cv2.waitKey(0)
