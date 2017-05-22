import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from pipeline import process, display_lines, draw_lines

from moviepy.editor import VideoFileClip
from IPython.display import HTML
from collections import deque


from undistort import get_undistorter
from visualize import draw_images as di

from pipeline import put_text


# binary filter methods

def norm(s):
    """normalizes and image"""
    return 255 * np.absolute(s) / np.max(s)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def mask(zeros, yl=230, yr=1050):
    """masks a binary for the area of interest marked by yl (y_right) and yr (y_right)"""
    zeros[:, :yl] = 0
    zeros[:, yr:] = 0
    return zeros


def sobel(gray, x=0, norm=True, **kwargs):
    """returns sobel gradients in x or y"""
    # gray = grayscale(img)
    s = np.absolute(cv2.Sobel(gray, cv2.CV_64F, x, abs(x - 1), **kwargs))
    if norm:
        s = 255 * np.absolute(s) / np.max(s)
    return s


def thresh(s, mi, mx, zeros=None):
    """applies thresholds to binary"""
    if zeros is None:
        zeros = np.zeros_like(s)
    zeros[(s >= mi) & (s <= mx)] = 1
    return zeros


def R_thresh(img, lim=(150, 255)):
    """filters for Red channel thresholds"""
    R = img[:, :, 0]
    return thresh(R, *lim)


def abs_thresh(img, mix, mxx, miy, mxy):
    """filters via thresholds for absolute sobel gradients in x and y"""
    sobx = sobel(img, 1)
    soby = sobel(img)
    return thresh(soby, miy, mxy, thresh(sobx, mix, mxx))


def mag_thresh(img, mi, mx, ksize=3):
    """filters via sobel magnitude thresholds"""
    sobx = sobel(img, 1, False, ksize=ksize)
    soby = sobel(img, 0, False, ksize=ksize)
    sob_mag = np.sqrt(sobx**2 + soby**2)
    sob_mag = np.uint8(255 * sob_mag / np.max(sob_mag))
    return thresh(sob_mag, mi, mx)


def grad_thresh(img, mi, mx, ksize=3):
    """filters via sobel gradient thresholds"""
    sobx = sobel(img, 1, False, ksize=ksize)
    soby = sobel(img, 0, False, ksize=ksize)
    grad = np.arctan2(np.absolute(soby), np.absolute(sobx))
    return thresh(grad, mi, mx)


def combine(*args):
    """combines binaries"""
    res = args[0]
    for arg in args[1:]:
        res[(arg == 0)] = 0
    return res


def alt_combine(*args, **kwargs):
    """combines binaries. can perform both AND and OR operations"""
    do_or = kwargs.get("do_or", False)
    res = args[0]
    val = 1 if do_or else 0
    for arg in args[1:]:
        res[(arg == val)] = val
    return res


def get_hls_thresh(img, h_thresh=None, s_thresh=None):
    """provides H and S channel thresholds to filter binaries"""
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    S = hls[:, :, 2]
    thresh = (15, 100)
    binary = np.zeros_like(H)
    if h_thresh:
        if s_thresh:
            binary[(H > h_thresh[0]) & (H <= h_thresh[1]) & (S > s_thresh[0]) & (S <= s_thresh[1])] = 1
        else:
            binary[(H > h_thresh[0]) & (H <= h_thresh[1])] = 1
    elif s_thresh:
        binary[(S > s_thresh[0]) & (S <= s_thresh[1])] = 1

    # binary[(H > h_thresh[0]) & (H <= h_thresh[1]) & (S > s_thresh[0]) & (S <= s_thresh[1])] = 1
    return binary  # combine(mask_H, mask_S)

# ends binary filter methods


# perspective transform method
rendered = False
p_matrix = None
p_inv_matrix = None


def pers_transform(image, reverse=False):
    """performs perspective tranformation"""
    global p_matrix, p_inv_matrix, rendered
    if not rendered:
        src = np.float32([[561, 474], [725, 474], [1040, 677], [254, 677]])
        xr = (1040 + 725)//2
        xl = (254 + 561)//2
        dst = np.float32([[xl, 0], [xr, 0], [xr, 720], [xl, 720]])
        p_matrix = cv2.getPerspectiveTransform(src, dst)
        p_inv_matrix = cv2.getPerspectiveTransform(dst, src)
        rendered = True
    m = p_matrix if (not reverse) else p_inv_matrix
    return cv2.warpPerspective(image, m, (image.shape[:2][::-1]), flags=cv2.INTER_LINEAR)


# testing code

undistorter = get_undistorter()  # debug with: lambda x: x


def test_images(bug=False):
    """for testing and debugging hyperparams"""
    folder = "bugs" if bug else "test_images"
    images = glob.glob("{}/*.jpg".format(folder))  # bug_img_2

    # ori_images = [undistorter(mpimg.imread(image)) for image in images]

    # count = 0
    imgs = []
    for fimg in images:
        image = mpimg.imread(fimg)
        img = undistorter(image)
        image = np.copy(img)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # gaussian blur
        img = cv2.GaussianBlur(img, (5, 5), 0)

        abs_args = (gray, 35, 120, 15, 40)  # (img, 35, 120, 30, 40)
        mag_args = (gray, 10, 255, 9)  # (img, 10, 200, 9)
        grad_args = (gray, 0.7, 1.3, 9)

        # white right : hight H, low S
        h_thresh_white = (100, 130)  # (100, 140) # (110, 150) (15, 90)(0, 90)
        s_thresh_white = (0, 15)  # (0, 20) # (0, 30) (90, 255)

        h_thresh_white_2 = (0, 90)  # (0, 30)
        s_thresh_white_2 = (90, 255)

        ans = alt_combine(abs_thresh(*abs_args), alt_combine(mag_thresh(*mag_args),
                          grad_thresh(*grad_args)), do_or=True)

        hls_thresh = get_hls_thresh(image, h_thresh_white_2, s_thresh_white_2)

        hls_thresh = alt_combine(get_hls_thresh(image, h_thresh_white, s_thresh_white),
                                 get_hls_thresh(image, h_thresh_white_2, s_thresh_white_2),
                                 do_or=True)

        ans = alt_combine(ans, hls_thresh)

        ans = pers_transform(ans)

        # drawing lines
        (left_fit, right_fit, measures) = process(ans)
        image = draw_lines(ans, image, left_fit, right_fit, pers_transform)
        put_text(image, measures)
        imgs.append(image)  # append(ans)

    di(imgs)


# Converts a clip from raw file to a file with lane overlay
def write_clip(input_file, output_file, function):
    clip = VideoFileClip(input_file)

    global left_line_points, right_line_points, fail_count, curvature_queue
    left_line_points = None
    right_line_points = None
    fail_count = 0
    curvature_queue = deque([])

    white_clip = clip.fl_image(function)  # NOTE: this function expects color images!
    white_clip.write_videofile(output_file, audio=False)


# global
count = 0


def process_frame(image, debug=False):
    """processes each frame provided and draws the lane lines, also writing measurements"""
    copy = np.copy(image)
    copy = undistorter(copy)

    gray = cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY)

    # gaussian blur
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    abs_args = (gray, 35, 120, 15, 40)  # (img, 35, 120, 30, 40)
    mag_args = (gray, 10, 255, 9)  # (img, 10, 200, 9)
    grad_args = (gray, 0.7, 1.3, 9)

    # TODO add multiple h/s thresholds
    # h_thresh_white = (100, 130)  # (100, 140) # (110, 150) (15, 90)(0, 90)
    # s_thresh_white = (0, 15)  # (0, 20) # (0, 30) (90, 255)

    h_thresh_white_2 = (0, 90)  # (0, 30)
    s_thresh_white_2 = (90, 255)
    # TODO OR combination between magnitude/gradient and x/y sobels
    ans = alt_combine(abs_thresh(*abs_args), alt_combine(mag_thresh(*mag_args),
                                                         grad_thresh(*grad_args)), do_or=True)

    hls_thresh = get_hls_thresh(copy, h_thresh_white_2, s_thresh_white_2)
    ans = alt_combine(ans, hls_thresh)

    # deprecated. remove mask for tough curves
    # ans = mask(ans)

    if debug:
        global count
        count += 1
        ans = pers_transform(ans)
        # uncomment to visualize
        image[(ans == 1), :] = 0
        image = np.dstack((ans, ans, ans))
        image[(ans == 1), :] = 255
        if count >= 40:
            pass
    else:
        ans = pers_transform(ans)
        (left_fit, right_fit, measures) = process(ans)
        image = draw_lines(ans, image, left_fit, right_fit, pers_transform)
        put_text(image, measures)

    return image


def test(debug=False, challenge=False):
    """set debug to try on smaller videos"""
    vinp = 'project_video.mp4'
    vout = 'project_video_solution.mp4'
    if debug:
        vinp = "project_video_error_short.mp4"
        vout = "project_video_error_solution_test.mp4"

    if challenge:
        vinp = 'challenge_video.mp4'
        vout = 'challenge_video_solution.mp4'

    write_clip(vinp, vout, process_frame)


if __name__ == "__main__":
    test()
    # test_images()

