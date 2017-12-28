import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
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
# Load the intrinsic parameter of the camera
def load_camera_parameters(path_camera):
    with open(path_camera + 'dist.pickle', 'rb') as handle:
        dist = np.array(pickle.load(handle))
    with open(path_camera + 'mtx.pickle', 'rb') as handle:
        mtx = np.array(pickle.load(handle))
    return dist, mtx
# ------------------------------------------------------------------------------------
# Remove distortion from the image
def undistort_image(image, dist, mtx):
    return cv2.undistort(image, mtx, dist, None, mtx)
# ------------------------------------------------------------------------------------
# Create the IPM image view
def ipm_image_proc(image, roi):

    # color = [0, 255, 255]
    # w = 2
    # cv2.line(image, tuple(roi[0]), tuple(roi[1]), color, w)
    # cv2.line(image, tuple(roi[1]), tuple(roi[2]), color, w)
    # cv2.line(image, tuple(roi[2]), tuple(roi[3]), color, w)
    # cv2.line(image, tuple(roi[3]), tuple(roi[0]), color, w)
    # cv2.imshow("image image", image)
    # cv2.waitKey(0)

    src = np.float32(roi)
    offset = 50
    w = image.shape[0]
    h = image.shape[1] - 500
    img_out_size = (w, h)

    dst = np.float32([[offset, 0], [img_out_size[0] - offset, 0], [img_out_size[0] - offset, img_out_size[1]], [offset, img_out_size[1]]])

    H = cv2.getPerspectiveTransform(src, dst)
    H_inv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(image, H, img_out_size)

    return warped, H, H_inv
# ------------------------------------------------------------------------------------
def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level + 1) * height):int(img_ref.shape[0] - level * height),
    max(0, int(center - width / 2)):min(int(center + width / 2), img_ref.shape[1])] = 1
    return output
# ------------------------------------------------------------------------------------
def find_window_centroids(image, window_width, window_height, margin):
    window_centroids = []  # Store the (left,right) window centroid positions per level
    window = np.ones(window_width)  # Create our window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 6):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 6):, int(image.shape[1] / 2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width / 2 + int(image.shape[1] / 2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(image.shape[0] / window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(
            image[int(image.shape[0] - (level + 1) * window_height):int(image.shape[0] - level * window_height), :],
            axis=0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
        offset = window_width / 2
        l_min_index = int(max(l_center + offset - margin, 0))
        l_max_index = int(min(l_center + offset + margin, image.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index]) + l_min_index - offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - margin, 0))
        r_max_index = int(min(r_center + offset + margin, image.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index]) + r_min_index - offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids
# ------------------------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------------------------
# Slice Window Parameters
# ------------------------------------------------------------------------------------
window_width = 50
window_height = 80  # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching
# ------------------------------------------------------------------------------------
# IPM (Inverse Perspective Mapping) Parameters
# ------------------------------------------------------------------------------------
#          left_top ****** right_top
#                  ********
#                 **********
#                ************
#               **************
#  left_bottom **************** right_bottom
# ------------------------------------------------------------------------------------
bottom = 680
top = 455

left_t = 565
right_t = 750

left_b = 190
right_b = 1230

left_top = [left_t, top]
right_top = [right_t, top]

left_bottom = [left_b, bottom]
right_bottom = [right_b, bottom]
# -------------------------------------------------------------------
roi = [left_top, right_top, right_bottom, left_bottom]
# -------------------------------------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------------------------------------
# Load camera parameters
path_camera='./camera_cal/'
dist, mtx = load_camera_parameters(path_camera)
# ------------------------------------------------------------------------------------
# Binary segmentation
path = './test_images/'
idx = 4
image = cv2.imread(path +'test' + str(idx) + '.jpg')
image = undistort_image(image, dist, mtx)
imshape = image.shape
# cv2.imshow("image hls_binary", image)
# cv2.waitKey(0)

hls_binary = hls_segmentation(image, thresh=(80, 255))
gradient_binary = gradient_segmentation(image, thresh=(30, 200))
img_segmentation = combine_segmentations(gradient_binary, hls_binary)*255

#img_segmentation= hls_binary*255
# cv2.imshow("image hls_binary", hls_binary*255)
# cv2.waitKey(0)
# cv2.imshow("image gradient_binary", gradient_binary*255)
# cv2.waitKey(0)
# cv2.imshow("image segmentation", img_segmentation)
# cv2.waitKey(0)
# ------------------------------------------------------------------------------------

ipm_img, H, H_inv = ipm_image_proc(img_segmentation, roi)
# -------------------------------------------------------------------
# cv2.imshow("image ipm_img", ipm_img)
# cv2.waitKey(0)

window_centroids = find_window_centroids(ipm_img, window_width, window_height, margin)

ipm_img_color = cv2.cvtColor(ipm_img,cv2.COLOR_GRAY2RGB)
color_l = [0, 255, 0]
color_r = [0, 0, 255]
if len(window_centroids) > 0:
   for level in range(0, len(window_centroids)):
       cv2.circle(ipm_img_color, (max(0, int(window_centroids[level][0] - (window_height / 2) + window_width )),int(ipm_img_color.shape[0] - (level + 1) * window_height )), 8, color_l,-1)
       cv2.circle(ipm_img_color, (max(0, int(window_centroids[level][1] - (window_height / 2) + window_width)),
                              int(ipm_img_color.shape[1] - (level + 1) * window_height)), 8, color_r, -1)

   cv2.imshow("points", ipm_img_color)
   cv2.waitKey(0)

if len(window_centroids) > 0:

    # Points used to draw all the left and right windows
    l_points = np.zeros_like(ipm_img)
    r_points = np.zeros_like(ipm_img)

    # Go through each level and draw the windows
    for level in range(0, len(window_centroids)):
        # Window_mask is a function to draw window areas
        l_mask = window_mask(window_width, window_height, ipm_img, window_centroids[level][0], level)
        r_mask = window_mask(window_width, window_height, ipm_img, window_centroids[level][1], level)
        # Add graphic points from window mask here to total pixels found
        l_points[(l_points == 255) | ((l_mask == 1))] = 255
        r_points[(r_points == 255) | ((r_mask == 1))] = 255

    # Draw the results
    template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
    zero_channel = np.zeros_like(template)  # create a zero color channel
    template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
    warpage = np.dstack((ipm_img, ipm_img, ipm_img)) * 255  # making the original road pixels 3 color channels
    output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

# If no window centers found, just display orginal road image
else:
    output = np.array(cv2.merge((ipm_img, ipm_img, ipm_img)), np.uint8)

# Display the final results
plt.imshow(output)
plt.title('window fitting results')
plt.show()

