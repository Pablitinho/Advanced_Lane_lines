## Writeup Advanced Lane Finding


---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/detection_1.JPG "Detection 1"
[image2]: ./output_images/detection_2.JPG "Detection 2"
[image3]: ./output_images/detection_3.JPG "Detection 3"
[image4]: ./output_images/Undistort.JPG "Undistorted image"
[image5]: ./output_images/Undistort_road.JPG "Undistorted road"
[image6]: ./output_images/Map_of_attention.png "Map of attention"
[image7]: ./output_images/IPM.png "IPM"
[image8]: ./output_images/derivate_lines_road.PNG "Derivate in X on the road"
[image9]: ./output_images/binary_segmentation.JPG "IPM"
[image10]: ./output_images/Lane_detection_diagram.png "Lane Detection Flow"
[image11]: ./output_images/accumulation.png "Accumulation"
[image12]: ./output_images/line_fitting.png "Line Fitting"
[image13]: ./output_images/Example_lane.png "Lane detection example"


[video1]: ./output.avi "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file camera_calibration.py

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

The first step is define the chessboard, defining the number of inner corners (in our case 9x6).  We apply the conversion to gray and detect the points as shown below:

![alt text][image1]
![alt text][image2]
![alt text][image3]

We collect all the points of each board and estimate the camera intrinsic parameters by mean of the function `cv2.calibrateCamera()`. This function return the extrinsic parameters like optical center, focal distance, etc and also the distorsion parameters. In order to correct the aberration produced by the len distorsion, we apply the function `cv2.undistort()` to rectify the image like it is shown below:

![alt text][image4]

### Pipeline (single images)

The global pipeline is described in the figure shown below:

![alt text][image10]

#### 1. Provide an example of a distortion-corrected image.

Below you can see how the original image (left) is distorted (right) making use of the camara parameters obtained in the previos section:
![alt text][image5]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In order to create the binary image, I use the combination of gray scale image (to detect the white color) and the saturation of the HSV color space like this: 

map_of_attention = Gray_scale + Saturation_channel

The first image is the original one, the second is the smoothed one and the last one belong to the map of attention:
![alt text][image6]

####NOTE: The binary segmentation is explained in the section 3

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The part of the code that transform the image into IPM (Bird-eye view) is ` ipm_image_proc()`. This function take the coordinates as show below from the image and transform into the new coordinates of the image:
# ------------------------------------------------------------------------------------
# IPM (Inverse Perspective Mapping) Parameters
# ------------------------------------------------------------------------------------
#              left_top ****** right_top
#                          ********
#                         **********
#                       ************
#                      **************
# left_bottom **************** right_bottom
# ------------------------------------------------------------------------------------
To obtain the transformation matrix from the original image to the new image (IPM) we make use of this opencv function:

H = cv2.getPerspectiveTransform(src, dst)

Where source is the points on the road in the image and destination are the corner points of the image.

After obtain this transformation matrix, it is possible to transform the original image to the bird-eye view by mean of opencv function: 

warped = cv2.warpPerspective(image, H, img_out_size)

Here it is an example of the original input to the IPM view.

![alt text][image7]

In this point is when we apply the binary segmentation by mean of the technique proposed in the the paper "Adaptative Road Lanes Detection and Classification" 

![alt text][image8]

After apply the segmentation described in this paper we obtain: 

![alt text][image9]

In order to get more details in the lines with no contiguous segment we applied the integration over the time as described this formular:

    road_features_integral=road_features_integral*(1-alfa)+road_features*alfa
    road_features_integral[(road_features_integral < 40)] = 0
    road_features_integral[(road_features_integral > 255)] = 255

This equation allow us to create the no contiguous segments into a one (like) single line. 

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

At first the Y position where are the lines must be identified. To find the y position we will accumulate the value in the Y position, as show in the image below:

![alt text][image11]

Once that we have the accumulation, we look from the middle to the left and the middle to the right the high value to search the shape of the line. For the left and right candidate position we generate blocks and estimate the median value positions. Once it is estimated the median position value, we create another block on the top of the other one and compute the median value, and so on... In this way we will get a position of each block always that there are a minimum number of pixels within the block. 

At this point we are able to compute the line by mean of a polynomial of second order. One example of the line fitting of the left and right line is shown below:

![alt text][image12]


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

To determinate the radius of the curvature I was following the "Measuring module in this project" and I did the estimation as follow:

    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    left_fit_m = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
    right_fit_m = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)

    left_curverad = ((1 + (2 * left_fit_m[0] * y_eval * ym_per_pix + left_fit_m[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit_m[0])
    right_curverad = ((1 + (2 * right_fit_m[0] * y_eval * ym_per_pix + right_fit_m[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit_m[0])

    print(left_curverad, right_curverad)

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.


I implemented this step with a function called draw_lane(). Here is an example of the output

![alt text][image13]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output.avi)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
