import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_images(path,num_images):

    image_list = []
    for idx in range(1,num_images):
        #image = np.asarray(Image.open(path+str(idx)+'.jpg')
        image = np.asarray(cv2.imread(path + str(idx) + '.jpg'))
        image_list.append(image)

    return np.array(image_list)
# ------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------
# Load Images
path = './camera_cal/'
image_list = load_images(path+'calibration', 20)

# Generate the points of the chessboard and empty detected points
obj_points = []
img_points = []
objp = np.zeros((6*9, 3),np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Detect corners in the images and collect them
for idx in range(image_list.shape[0]):
    gray = cv2.cvtColor(image_list[idx], cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
    if ret:
        print("Image: ", idx)
        img_points.append(corners)
        obj_points.append(objp)
        # Plot the detected points
        # img = cv2.drawChessboardCorners(image_list[idx], (9, 6), corners, ret)
        # plt.imshow(img)
        # plt.show()

# Calibrate the Camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Undistort an image
img_undistort = cv2.undistort(image_list[0], mtx, dist, None, mtx)
f, axarr = plt.subplots(1,2)
axarr[0].imshow(image_list[0])
axarr[1].imshow(img_undistort)
plt.show()

# Save the information of the intrinsic camera calibration
if ret:
    with open(path+'mtx.pickle', 'wb') as handle:
        pickle.dump(mtx, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path+'dist.pickle', 'wb') as handle:
        pickle.dump(dist, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path+'rvecs.pickle', 'wb') as handle:
        pickle.dump(rvecs, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path+'tvecs.pickle', 'wb') as handle:
        pickle.dump(tvecs, handle, protocol=pickle.HIGHEST_PROTOCOL)

