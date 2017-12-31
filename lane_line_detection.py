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
    return binary_output, s_channel
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
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, :int(image.shape[1] / 2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width / 2
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, int(image.shape[1] / 2):], axis=0)
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
def findLines(image, nwindows=9, margin=110, minpix=50):
    """
    Find the polynomial representation of the lines in the `image` using:
    - `nwindows` as the number of windows.
    - `margin` as the windows margin.
    - `minpix` as minimum number of pixes found to recenter the window.
    - `ym_per_pix` meters per pixel on Y.
    - `xm_per_pix` meters per pixels on X.

    Returns (left_fit, right_fit, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)
    """
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    # Make a binary and transform image
    binary_warped = image
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
        nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.median(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.median(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Fit a second order polynomial to each
    left_fit_m = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_m = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    return (left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy)
# ------------------------------------------------------------------------------------
def showLaneOnImages(image, cols = 2, rows = 3, figsize=(15,13)):
    """
    Display `images` on a [`cols`, `rows`] subplot grid.
    Returns a collection with the image paths and the left and right polynomials.
    """
    #fig, axes = plt.subplots(1, 1, figsize=figsize)
    indexes = range(cols * rows)
    imageAndFit = []
    #ax = axes.flat

    left_fit, right_fit, left_fit_m, right_fit_m = visualizeLanes(image, plt)
    #plt.axis('off')
    imageAndFit.append((left_fit, right_fit, left_fit_m, right_fit_m ) )

    return imageAndFit
def visualizeLanes(image, ax):
    """
    Visualize the windows and fitted lines for `image`.
    Returns (`left_fit` and `right_fit`)
    """
    left_fit, right_fit, left_fit_m, right_fit_m, left_lane_inds, right_lane_inds, out_img, nonzerox, nonzeroy = findLines(image)
    # Visualization
    ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    ax.imshow(out_img)
    ax.plot(left_fitx, ploty, color='yellow')
    ax.plot(right_fitx, ploty, color='yellow')
    ax.pause(0.05)
    ax.clf()
    return ( left_fit, right_fit, left_fit_m, right_fit_m )

def get_road_features(s_channel):

    thr = 35
    max_width = 30
    binary_image = np.uint8(np.zeros(s_channel.shape))

    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0) # Take the derivative in x

    mark_positive = False
    mark_negative = False
    id_x_positive = -1
    id_x_negative = -1
    id_x_cnt = -1
    for id_y in range(1, s_channel.shape[0]-1):
        for id_x in range(1, s_channel.shape[1]-1):

            if (sobelx[id_y,id_x]>thr and sobelx[id_y,id_x-1]<sobelx[id_y,id_x] and sobelx[id_y,id_x+1]<sobelx[id_y,id_x]):
                dx = (s_channel[id_y,id_x-1] - s_channel[id_y,id_x-1])/2
                dy = (s_channel[id_y + 1, id_x] - s_channel[id_y-1, id_x]) / 2
                absgraddir = np.arctan2(np.absolute(dy), np.absolute(dx))
                #if (absgraddir<1.6):
               # binary_image[id_y, id_x] = 255
                mark_positive = True
                mark_negative = False
                id_x_positive = id_x
                id_x_cnt = 0
            # Count the width of the candidate if local maxima is detected
            if mark_positive == True:
                id_x_cnt += 1
            # If the width of the candidate is bigger than the max_width, the local maxima is reseted
            if id_x_cnt > max_width:
                id_x_cnt = -1
                id_x_positive = -1
                mark_positive = False
                mark_negative = False
                id_x_negative = -1

            if (sobelx[id_y,id_x]<-thr and sobelx[id_y,id_x-1]>sobelx[id_y,id_x] and sobelx[id_y,id_x+1]>sobelx[id_y,id_x]):
                dx = (s_channel[id_y,id_x-1] - s_channel[id_y,id_x-1])/2
                dy = (s_channel[id_y + 1, id_x] - s_channel[id_y-1, id_x]) / 2
                absgraddir = np.arctan2(np.absolute(dy), np.absolute(dx))
                #if (absgraddir < 1.6):
                #binary_image[id_y, id_x] = 125

                if (mark_positive == True and id_x_cnt > 5):
                    mark_negative = True
                    id_x_negative = id_x
                if (mark_positive == True and mark_negative == True and id_x_cnt > 0 and id_x_positive > 0):
                    for id_x_marker in range(id_x_cnt):
                        if (id_x_positive + id_x_marker<s_channel.shape[1]-1 and binary_image[id_y, id_x_positive + id_x_marker] == 0):
                           binary_image[id_y, id_x_positive + id_x_marker] = 255

                    id_x_positive = -1
                    mark_positive = False
                    mark_negative = False
                    id_x_cnt = -1

        id_x_positive = -1
        mark_positive = False
        mark_negative = False
        id_x_cnt = -1
    # cv2.imshow("satur", binary_image)
    # cv2.waitKey(0)
    return binary_image
# ------------------------------------------------------------------------------------
def plot_lane_dots(binary, window_centroids, window_width, window_height):

    ipm_img_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    color_l = [0, 255, 0]
    color_r = [0, 0, 255]
    if len(window_centroids) > 0:
        for level in range(0, len(window_centroids)):
            cv2.circle(ipm_img_color, (max(0, int(window_centroids[level][0] - (window_height / 2) + window_width/2)),
                                       int(ipm_img_color.shape[0] - (level + 1) * window_height)), 8, color_l, -1)
            cv2.circle(ipm_img_color, (max(0, int(window_centroids[level][1] - (window_height / 2) + window_width/2)),
                                       int(ipm_img_color.shape[1] - (level + 1) * window_height)), 8, color_r, -1)

        cv2.imshow("points", ipm_img_color)
        cv2.waitKey(10)

    # if len(window_centroids) > 0:
    #
    #     # Points used to draw all the left and right windows
    #     l_points = np.zeros_like(ipm_img)
    #     r_points = np.zeros_like(ipm_img)
    #
    #     # Go through each level and draw the windows
    #     for level in range(0, len(window_centroids)):
    #         # Window_mask is a function to draw window areas
    #         l_mask = window_mask(window_width, window_height, ipm_img, window_centroids[level][0], level)
    #         r_mask = window_mask(window_width, window_height, ipm_img, window_centroids[level][1], level)
    #         # Add graphic points from window mask here to total pixels found
    #         l_points[(l_points == 255) | ((l_mask == 1))] = 255
    #         r_points[(r_points == 255) | ((r_mask == 1))] = 255
    #
    #     # Draw the results
    #     template = np.array(r_points + l_points, np.uint8)  # add both left and right window pixels together
    #     zero_channel = np.zeros_like(template)  # create a zero color channel
    #     template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
    #     warpage = np.dstack((ipm_img, ipm_img, ipm_img)) * 255  # making the original road pixels 3 color channels
    #     output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)  # overlay the orignal road image with window results

    # If no window centers found, just display orginal road image
    else:
        output = np.array(cv2.merge((ipm_img, ipm_img, ipm_img)), np.uint8)

    # Display the final results
    # plt.imshow(output)
    # plt.title('window fitting results')
    # plt.show()
# ------------------------------------------------------------------------------------
def generate_polynom(road_features, window_centroids):

    x_left = []
    y_left = []
    x_right = []
    y_right = []
    if len(window_centroids) > 0:
       for level in range(0, len(window_centroids)):
           y_left.append(max(0, int(window_centroids[level][0] - (window_height / 2) + window_width)))
           x_left.append(int(road_features.shape[0] - (level + 1) * window_height))

           y_right.append(max(0, int(window_centroids[level][1] - (window_height / 2) + window_width)))
           x_right.append(int(road_features.shape[1] - (level + 1) * window_height))

    left_fit  = np.polyfit(x_left, y_left, 2)
    right_fit = np.polyfit(x_right, y_right, 2)
    x_left = np.array(x_left)
    y_left = np.array(y_left)
    x_right = np.array(x_right)
    y_right = np.array(y_right)

    left_fity = left_fit[0] * x_left ** 2 + left_fit[1] * x_left + left_fit[2]
    right_fity = right_fit[0] * x_right ** 2 + right_fit[1] * x_right + right_fit[2]

    road_features = cv2.cvtColor(road_features, cv2.COLOR_GRAY2RGB)
    color_l = [0, 255, 0]
    color_r = [0, 0, 255]

    for x in range(x_left.shape[0]-1):
        #cv2.line(road_features,(np.int32(x_left[x]),np.int32(left_fity[x])),(np.int32(x_left[x+1]),np.int32(left_fity[x+1])),color_l)
        # Left
        cv2.line(road_features,(np.int32(left_fity[x]),np.int32(x_left[x])),(np.int32(left_fity[x+1]),np.int32(x_left[x+1])),color_l)
        # Right
        cv2.line(road_features,(np.int32(right_fity[x]),np.int32(x_right[x])),(np.int32(right_fity[x+1]),np.int32(x_right[x+1])),color_r)

    cv2.imshow("line", road_features)
    cv2.waitKey(10)

    return left_fit, right_fit, x_left, x_right
# ------------------------------------------------------------------------------------
# PARAMETERS
# ------------------------------------------------------------------------------------
# Slice Window Parameters
# ------------------------------------------------------------------------------------
window_width = 50
window_height = 50  # Break image into 9 vertical layers since image height is 720
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
idx = 1
road_features_integral=[]
cap = cv2.VideoCapture('project_video.mp4')

first_time = True
if (cap.isOpened()== False):
  print("Error opening video file")
  exit(-1)
while(cap.isOpened()):
    ret, image = cap.read()

    #image = cv2.imread(path +'test' + str(idx) + '.jpg')
    image = undistort_image(image, dist, mtx)
    g_kernel = cv2.getGaussianKernel(7, 3)
    image = cv2.filter2D(image, -1, g_kernel)

    imshape = image.shape
    # cv2.imshow("image hls_binary", image)
    # cv2.waitKey(0)

    hls_binary, s_channel = hls_segmentation(image, thresh=(80, 255))
    gradient_binary = gradient_segmentation(image, thresh=(30, 200))
    img_segmentation = combine_segmentations(gradient_binary, hls_binary)*255

    s_channel = np.float32(s_channel)

    gray2 = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    gray = (np.float32(image[:, :, 0]) + np.float32(image[:, :, 1]) + np.float32(image[:, :, 2]))/3
    yellow_image= ((255 -np.float32( image[:, :, 0]) + np.float32(image[:, :, 1]) + np.float32(image[:, :, 2]))/3)
    #img_segmentation = (img_segmentation+yellow_image*4)
    #img_augmented = (np.float32(gray)+np.float32(s_channel))
    img_augmented = np.float32((np.float32(gray2)+np.float32(s_channel)))
    # cv2.imshow("image segmentation",  np.uint8(img_segmentation))
    # cv2.waitKey(0)
    #img_segmentation= hls_binary*255
    # cv2.imshow("image hls_binary", hls_binary*255)
    # cv2.waitKey(0)
    # cv2.imshow("image gradient_binary", gradient_binary*255)
    # cv2.waitKey(0)
    # cv2.imshow("image segmentation", img_segmentation)
    # cv2.waitKey(0)
    # ------------------------------------------------------------------------------------
    ipm_img, H, H_inv = ipm_image_proc(img_augmented, roi)

    # cv2.line(img_augmented,(0,500),(img_augmented.shape[1],500),(255,0,0))
    # cv2.imshow("ipm", np.uint8(img_augmented))
    # cv2.waitKey(0)
    #

    import pylab
    # x = range(img_augmented.shape[1])
    # y = []
    #
    # y = np.float32(gray[499,:])
    # print(y)
    # plt.plot(x, y)
    # plt.show()
    # ipm_img_image, H, H_inv = ipm_image_proc(image[:,:,1], roi)

    road_features = get_road_features(ipm_img)
    #ipm_img = img_segmentation
    if first_time == True:
        first_time=False
        road_features_integral= road_features

    alfa = 0.2
    road_features_integral=road_features_integral*(1-alfa)+road_features*alfa
    #road_features_integral = ipm_img * (1 - alfa) + ipm_img * alfa

    road_features_integral[(road_features_integral < 40)] = 0
    road_features_integral[(road_features_integral > 255)] = 255

    # cv2.imshow("ipm", np.uint8(road_features_integral))
    # cv2.waitKey(20)
    # -------------------------------------------------------------------
    # cv2.imshow("image ipm_img", np.uint8(ipm_img_image))
    # cv2.waitKey(0)

    window_centroids = find_window_centroids(ipm_img, window_width, window_height, margin)

    #plot_lane_dots(np.uint8(road_features_integral), window_centroids, window_width, window_height)

    #left_fit, right_fit, x_left, x_right = generate_polynom(np.uint8(road_features_integral), window_centroids)

    imagesPoly = showLaneOnImages(np.uint8(road_features_integral))


