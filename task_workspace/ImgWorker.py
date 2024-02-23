import cv2
from ToGray import ToGray
import glob as gl
import numpy as np

class ImgWorker:
    def save_img(imgs, name):
        for i in range(imgs.shape[0]):
            cv2.imwrite('output_images/' + str(name) + "_" +str(i) +'.png', imgs[i])

    def smooth_resize(imgs, px):
        ret_data = []
        for i in range(imgs.shape[0]):
            ret_data.append(cv2.GaussianBlur(cv2.resize(imgs[i],(px, px)),(5,5),0))
        return np.asarray(ret_data)
        

    def load_img(filenames, px):
        #Reading input images
        data = []
        for file in filenames:
            img = cv2.resize(cv2.imread(file), (px, px))
            data.append(np.array(img)[:,:,:3])
        return data
    
    def to_grad_x(imgs):
        ret_data = []
        for i in range(imgs.shape[0]):
            ret_data.append(cv2.Sobel(imgs[i], ddepth=cv2.CV_32F, dx=1, dy=0))
        return np.asarray(ret_data)
    
    def to_grad_y(imgs):
        ret_data = []
        for i in range(imgs.shape[0]):
            ret_data.append(cv2.Sobel(imgs[i], ddepth=cv2.CV_32F, dx=0, dy=1))
        return np.asarray(ret_data)

