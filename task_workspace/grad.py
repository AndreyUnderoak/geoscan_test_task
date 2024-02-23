import cv2
from ToGray import ToGray
import glob as gl


path = "./input_images/*.png"

# count = 0
filenames = gl.glob(path)

tg = ToGray("./pt/ColorToGray.ckpt-49", filenames)

# image = cv2.imread("input_images/gray_100.png")


# gX = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=1, dy=0)
# gY = cv2.Sobel(image, ddepth=cv2.CV_32F, dx=0, dy=1)

# # show our output images
# cv2.imshow("image", image)
# cv2.imshow("Sobel/Scharr X", gX)
# cv2.imshow("Sobel/Scharr Y", gY)

# cv2.waitKey(0)