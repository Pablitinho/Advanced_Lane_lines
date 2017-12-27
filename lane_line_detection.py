import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
# ------------------------------------------------------------------------------------
def hls_segmentation(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    # 2) Apply a threshold to the S channel
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output
# ------------------------------------------------------------------------------------
def gradient_segmentation(img, thresh=(0, 255)):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return sxbinary
# ------------------------------------------------------------------------------------
def combine_segmentations(sxbinary, s_binary):
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary
# ------------------------------------------------------------------------------------
def load_camera_parameters(path_camera):
    with open(path_camera + 'dist.pickle', 'rb') as handle:
        dist = np.array(pickle.load(handle))
    with open(path_camera + 'mtx.pickle', 'rb') as handle:
        mtx = np.array(pickle.load(handle))
    return dist, mtx
# ------------------------------------------------------------------------------------
def undistort_image(image, dist, mtx):
    return cv2.undistort(image, mtx, dist, None, mtx)
# ------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------
# Load camera parameters
path_camera='./camera_cal/'
dist, mtx = load_camera_parameters(path_camera)

# ------------------------------------------------------------------------------------
# Binary segmentation
path = './test_images/'
idx = 1
image = cv2.imread(path +'test' + str(idx) + '.jpg')
image = undistort_image(image, dist, mtx)
imshape = image.shape
# cv2.imshow("image hls_binary", image)
# cv2.waitKey(0)

hls_binary = hls_segmentation(image, thresh=(80, 255))
gradient_binary = gradient_segmentation(image, thresh=(30, 200))
img_segmentation = combine_segmentations(gradient_binary, hls_binary)*255

# cv2.imshow("image hls_binary", hls_binary*255)
# cv2.waitKey(0)
# cv2.imshow("image gradient_binary", gradient_binary*255)
# cv2.waitKey(0)
# cv2.imshow("image segmentation", img_segmentation)
# cv2.waitKey(0)
# ------------------------------------------------------------------------------------
# IPM (Inverse Perspective Mapping)
# ------------------------------------------------------------------------------------
src_points = np.float32([[450, 320], [490, 320], [0, imshape[0]], [imshape[1], imshape[0]]])

h, w = image.shape[:2]
dst_points = np.float32([[w, h], [w - w, h], [w, h - h], [w - w, h - h]])
# d) use cv2.getPerspectiveTransform() to get M, the transform matrix

bottomY = 680
topY = 455

left1 = (190, bottomY)
left1_x, left1_y = left1
left2 = (575, topY)
left2_x, left2_y = left2

right1 = (735, topY)
right1_x, right1_y = right1

right2 = (1230, bottomY)
right2_x, right2_y = right2

color = [0, 255, 255]
w = 2
cv2.line(image, left1, left2, color, w)
cv2.line(image, left2, right1, color, w)
cv2.line(image, right1, right2, color, w)
cv2.line(image, right2, left1, color, w)
cv2.imshow("image image", image)
cv2.waitKey(0)

src = np.float32([ [left2_x, left2_y],[right1_x, right1_y],[right2_x, right2_y],[left1_x, left1_y]])
#src = np.float32([ left1,left2,right1,right2])

offset = 50
nX = image.shape[0]
nY = image.shape[1]-400
img_size = (nX, nY)
dst = np.float32([[offset, 0],[img_size[0]-offset, 0],[img_size[0]-offset, img_size[1]],[offset, img_size[1]]])

H = cv2.getPerspectiveTransform(src, dst)
H_inv = cv2.getPerspectiveTransform(dst, src)
warped = cv2.warpPerspective(image, H, img_size)

cv2.imshow("image warped", warped)
cv2.waitKey(0)