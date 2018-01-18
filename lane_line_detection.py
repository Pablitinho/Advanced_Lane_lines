import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import imageio as imio
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
def detect_lines(image, nwindows=9, margin=80, minpix=50):

    # Make a binary and transform image
    binary_warped = image

    # Compute the accumulation of the white pixels per each column
    Accumulator = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    # plt.plot(range(histogram.shape[0]),histogram, color='green')
    # plt.show()

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(Accumulator.shape[0] / 2)
    leftx_base = np.argmax(Accumulator[:midpoint])
    rightx_base = np.argmax(Accumulator[midpoint:]) + midpoint

    # Set height of windows
    window_height = np.int(binary_warped.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # List with the position of each block. If the block is empty, it will not be included into the list
    left_lane_pos = []
    right_lane_pos = []

    left_lane_num_points = 0
    right_lane_num_points = 0
    # Step through the windows one by one
    for window in range(nwindows):
        # Generate the block position and dimension
        window_y_bottom = binary_warped.shape[0] - (window + 1) * window_height
        windown_y_up = binary_warped.shape[0] - window * window_height
        win_x_left_bottom = leftx_current - margin
        win_x_left_up = leftx_current + margin

        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= window_y_bottom) & (nonzeroy < windown_y_up) & (nonzerox >= win_x_left_bottom) & (
        nonzerox < win_x_left_up)).nonzero()[0]
        good_right_inds = ((nonzeroy >= window_y_bottom) & (nonzeroy < windown_y_up) & (nonzerox >= win_xright_low) & (
        nonzerox < win_xright_high)).nonzero()[0]

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.median(nonzerox[good_left_inds]))
            left_lane_pos.append([leftx_current, (windown_y_up+window_y_bottom)/2])
            left_lane_num_points += len(good_left_inds)
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.median(nonzerox[good_right_inds]))
            right_lane_pos.append([rightx_current, (windown_y_up + window_y_bottom) / 2])
            right_lane_num_points += len(good_right_inds)

    # Fit a second order polynomial to each
    left_lane_pos = np.array(left_lane_pos)
    right_lane_pos = np.array(right_lane_pos)

    left_fit = np.polyfit(left_lane_pos[:,1], left_lane_pos[:,0], 2)
    right_fit = np.polyfit(right_lane_pos[:,1], right_lane_pos[:,0], 2)

    return left_fit, right_fit, left_lane_num_points, right_lane_num_points
# ------------------------------------------------------------------------------------
def draw_lane(img, left_fit, right_fit, Minv):

    yMax = img.shape[0]
    ploty = np.linspace(0, yMax - 1, yMax)
    color_warp = np.zeros_like(img).astype(np.uint8)

    # Calculate points.
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    warped_lane = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0]))
    return cv2.addWeighted(img, 1, warped_lane, 0.3, 0)
# ----------------------------------------------------------------------------------------------------------------
def estimate_curvature(yMax, left_fit_temporal,right_fit_temporal):
    # ----------------------------------------------------------------------
    ploty = np.linspace(0, yMax - 1, yMax)
    y_eval = np.max(ploty)

    left_fitx = left_fit_temporal[0] * ploty ** 2 + left_fit_temporal[1] * ploty + left_fit_temporal[2]
    right_fitx = right_fit_temporal[0] * ploty ** 2 + right_fit_temporal[1] * ploty + right_fit_temporal[2]

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    left_fit_m = np.polyfit(ploty * ym_per_pix, left_fitx * xm_per_pix, 2)
    right_fit_m = np.polyfit(ploty * ym_per_pix, right_fitx * xm_per_pix, 2)

    left_curverad = ((1 + (2 * left_fit_m[0] * y_eval * ym_per_pix + left_fit_m[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_m[0])
    right_curverad = ((1 + (2 * right_fit_m[0] * y_eval * ym_per_pix + right_fit_m[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_m[0])

    # ----------------------------------------------------------------------
    return left_curverad, right_curverad
# ----------------------------------------------------------------------------------------------------------------
def detect_lane(image, left_fit_temporal,right_fit_temporal,first_time):

    # Get the detected lines
    left_fit, right_fit, left_lane_num_points, right_lane_num_points = detect_lines(image)

    alfa = 0.3
    if first_time:
       left_fit_temporal = left_fit
       right_fit_temporal = right_fit
    else:
       left_fit_temporal = left_fit_temporal * (1 - alfa) + left_fit * alfa
       right_fit_temporal = right_fit_temporal * (1 - alfa) + right_fit * alfa

    # Get the curvature of both lines
    left_curverad, right_curverad = estimate_curvature(image.shape[0], left_fit_temporal, right_fit_temporal)

    if (left_lane_num_points>right_lane_num_points):
        curvature = left_curverad
    else:
        curvature = right_curverad

    print(curvature)

    return left_fit_temporal, right_fit_temporal, curvature
# --------------------------------------------------------------------------------
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
# PARAMETERS
# ------------------------------------------------------------------------------------
# Slice Window Parameters
# ------------------------------------------------------------------------------------
window_width = 50
window_height = 50  # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching
left_fit_temporal = []
right_fit_temporal=[]
first_time_line = True

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
#cap = cv2.VideoCapture('challenge_video.mp4')

first_time = True
if (cap.isOpened()== False):
  print("Error opening video file")
  exit(-1)
# cnt=0
# imio.plugins.ffmpeg.download()
size = (int(1280), int(720))
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'x264' doesn't work
out_video = cv2.VideoWriter('./output_curvature.avi', fourcc, 20.0, size, True)  # 'False' for 1-ch instead of 3-ch for color

while(cap.isOpened()):

    ret, image = cap.read()
    # ------------------------------------------------------------------------------------
    # Undistort the image
    # ------------------------------------------------------------------------------------
    image = undistort_image(image, dist, mtx)

    # ------------------------------------------------------------------------------------
    # Remove noise and smooth the signal
    # ------------------------------------------------------------------------------------
    g_kernel = cv2.getGaussianKernel(7, 3)
    image = cv2.filter2D(image, -1, g_kernel)

    # ------------------------------------------------------------------------------------
    # Combine the color spaces Saturation and Gray Scale (Yellow and white)
    # ------------------------------------------------------------------------------------
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:, :, 2]
    s_channel = np.float32(s_channel)
    gray2 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    map_of_attention = np.float32((np.float32(gray2)+np.float32(s_channel)))

    # ------------------------------------------------------------------------------------
    # Compute Birdeye view (IPM)
    # ------------------------------------------------------------------------------------
    ipm_img, H, H_inv = ipm_image_proc(map_of_attention, roi)

    # ------------------------------------------------------------------------------------
    # Get road features, Road_Features. Get the features by mean of Sobel x and detect the zero cross
    # ------------------------------------------------------------------------------------
    road_features = get_road_features(ipm_img)
    # cv2.imshow("image", road_features)
    # cv2.waitKey(0)
    # ------------------------------------------------------------------------------------
    # Temporal feature
    # ------------------------------------------------------------------------------------
    if first_time == True:
        first_time = False
        road_features_integral= road_features

    alfa = 0.2
    road_features_integral= road_features_integral*(1-alfa)+road_features*alfa

    road_features_integral[(road_features_integral < 40)] = 0
    road_features_integral[(road_features_integral > 255)] = 255
    # ------------------------------------------------------------------------------------
    # Detect lane
    # ------------------------------------------------------------------------------------
    left_fit_temporal, right_fit_temporal, curvature = detect_lane(np.uint8(road_features_integral), left_fit_temporal,
                                                        right_fit_temporal, first_time_line)
    first_time_line = False
    # ------------------------------------------------------------------------------------
    # Detect lane
    # ------------------------------------------------------------------------------------
    im = draw_lane(image, left_fit_temporal, right_fit_temporal, H_inv)

    cv2.putText(im, "Curvature:" + "{:.2f}".format(curvature), (20, 40), cv2.QT_FONT_NORMAL, 1, 255)
    cv2.imshow("Lane detection", im)
    cv2.waitKey(10)
    out_video.write(im)
