import cv2
import numpy as np

class ImgWorker:
    def save_img(imgs, name, postfix = ""):
        for i in range(imgs.shape[0]):
            # imk = imgs[i]
            # rows,cols = imk.shape
            # for i2 in range(rows):
            #     for j in range(cols):
            #         k = imk[i2,j]
            #         print(k)
            cv2.imwrite('output_images/' + str(name) + "_" +str(i) + str(postfix) +'.png', imgs[i])

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
            ret_data.append()
        return np.asarray(ret_data)

    def to_grad_x(imgs):
        ret_data = []
        for i in range(imgs.shape[0]):
            ret_data.append(cv2.Sobel(imgs[i], ddepth=cv2.CV_32F, dx=1, dy=0, borderType=cv2.BORDER_REFLECT_101))
        return np.asarray(ret_data)

    def to_grad_y(imgs):
        ret_data = []
        for i in range(imgs.shape[0]):
            ret_data.append(cv2.Sobel(imgs[i], ddepth=cv2.CV_32F, dx=0, dy=1, borderType=cv2.BORDER_REFLECT_101))
        return np.asarray(ret_data)

    def normalize(imgs):
        ret_data = []
        for i in range(imgs.shape[0]):
            ret_data.append(cv2.normalize(imgs[i], None, 0, 100.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F))
        return np.asarray(ret_data)
