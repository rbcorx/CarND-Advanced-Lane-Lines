import numpy as np
import cv2
import matplotlib.image as mtpimg
import glob


def get_calib_coef(objpts, imgpts, shape):
    """returns caliberations coefficients"""
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpts, imgpts, shape, None, None)
    return (ret, mtx, dist)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def get_calib_pts(images, size=(9, 6)):
    """extracts caliberation params from images (objpts, imgpts, shape)
    images must be array of filenames"""
    imgpts = []
    objpts = []
    objpts_img = np.zeros((size[0] * size[1], 3), np.float32)
    objpts_img[:, :2] = np.mgrid[0:size[0], 0:size[1]].T.reshape(-1, 2)
    for image_name in images:
        image = mtpimg.imread(image_name)
        gray = grayscale(image)
        ret, corners = cv2.findChessboardCorners(gray, size, None)
        if ret is True:
            imgpts.append(corners)
            objpts.append(objpts_img)
    shape = (image.shape[1], image.shape[0])
    return (objpts, imgpts, shape)


def get_undistorter():
    """returns the undistorter as a function after performing caliberation with the provided images"""
    images = glob.glob("camera_cal/calibration*.jpg")
    ret, mtx, dist = get_calib_coef(*get_calib_pts(images))

    def undistort(image):
        return cv2.undistort(image, mtx, dist, None, mtx)

    return undistort




