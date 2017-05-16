import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# 4:44

# binary filter methods


def norm(s):
    return 255 * np.absolute(s) / np.max(s)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def mask(zeros, yl=230, yr=1050):
    zeros[:, :yl] = 0
    zeros[:, yr:] = 0
    return zeros


def sobel(gray, x=0, norm=True, **kwargs):
    # gray = grayscale(img)
    s = np.absolute(cv2.Sobel(gray, cv2.CV_64F, x, abs(x - 1), **kwargs))
    if norm:
        s = 255 * np.absolute(s) / np.max(s)
        # print(s[:30])
    return s


def thresh(s, mi, mx, zeros=None):
    if zeros is None:
        zeros = np.zeros_like(s)
    zeros[(s >= mi) & (s <= mx)] = 1
    return zeros


def R_thresh(img, lim=(150, 255)):
    R = img[:, :, 0]
    return thresh(R, *lim)


def abs_thresh(img, mix, mxx, miy, mxy):
    sobx = sobel(img, 1)
    soby = sobel(img)
    return thresh(soby, miy, mxy, thresh(sobx, mix, mxx))


def mag_thresh(img, mi, mx, ksize=3):
    sobx = sobel(img, 1, False, ksize=ksize)
    soby = sobel(img, 0, False, ksize=ksize)
    sob_mag = np.sqrt(sobx**2 + soby**2)
    sob_mag = np.uint8(255 * sob_mag / np.max(sob_mag))
    return thresh(sob_mag, mi, mx)


def grad_thresh(img, mi, mx, ksize=3):
    sobx = sobel(img, 1, False, ksize=ksize)
    soby = sobel(img, 0, False, ksize=ksize)
    grad = np.arctan2(np.absolute(soby), np.absolute(sobx))
    return thresh(grad, mi, mx)


def combine(*args):
    res = args[0]
    for arg in args[1:]:
        res[(arg == 0)] = 0
    return res

def alt_combine(*args, **kwargs):
    do_or = kwargs.get("do_or", False)
    res = args[0]
    val = 1 if do_or else 0
    for arg in args[1:]:
        res[(arg == val)] = val
    return res

def get_hls_thresh(img, h_thresh=None, s_thresh=None):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    S = hls[:, :, 2]
    # mask_H = thresh(H, h_thresh[0], h_thresh[1])
    #mask_S = thresh(S, s_thresh[0], s_thresh[1])
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
    return binary #combine(mask_H, mask_S)
    """
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary_output
    """


# binary filter methods end

# perspective transform method

rendered = False
p_matrix = None
p_inv_matrix = None


def pers_transform(image, reverse=False):
    global p_matrix, p_inv_matrix, rendered
    if not rendered:
        src = np.float32([[561, 474], [725, 474], [1040, 677], [254, 677]])
        xr = (1040 + 725)//2
        xl = (254 + 561)//2
        # np.float32([[622, 435], [657, 435], [1040, 677], [254, 677]])
        dst = np.float32([[xl, 0], [xr, 0], [xr, 720], [xl, 720]])
        p_matrix = cv2.getPerspectiveTransform(src, dst)
        p_inv_matrix = cv2.getPerspectiveTransform(dst, src)
        rendered = True
    m = p_matrix if (not reverse) else p_inv_matrix
    return cv2.warpPerspective(image, m, (image.shape[:2][::-1]), flags=cv2.INTER_LINEAR)

# perspective transform ends


# testing code

from undistort import get_undistorter
from visualize import draw_images as di


undistorter = lambda x: x  # get_undistorter()


def test_images(bug=True):
    folder = "bugs" if bug else "test_images"
    images = glob.glob("{}/bug_img_2.jpg".format(folder))

    ori_images = [undistorter(mpimg.imread(image)) for image in images]

    count = 0
    imgs = []
    for fimg in images:

        img = undistorter(mpimg.imread(fimg))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        #img = pers_transform(img)

        # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        abs_args = (gray, 35, 120, 15, 40) # (img, 35, 120, 30, 40)
        mag_args = (gray, 10, 255, 9) # (img, 10, 200, 9)
        grad_args = (gray, 0.7, 1.3, 9)

        h_thresh = (0, 79)
        s_thresh = (130, 255)

        # white right : hight H, low S

        h_thresh_white = (100, 130)  # (100, 140) # (110, 150) (15, 90)(0, 90)
        s_thresh_white = (0, 15)  # (0, 20) # (0, 30) (90, 255)

        h_thresh_white_2 = (0, 90) # (0, 30)
        s_thresh_white_2 = (90, 255)
        # TODO OR combination between magnitude/gradient and x/y sobels
        ans = alt_combine(abs_thresh(*abs_args), alt_combine(mag_thresh(*mag_args), grad_thresh(*grad_args)), do_or=True)

        # R thresh:
        # r_t = R_thresh(img)
        # ans = alt_combine(r_t, ans)

        hls_thresh = get_hls_thresh(img, h_thresh_white_2, s_thresh_white_2)

        # hls_thresh = alt_combine(get_hls_thresh(img, h_thresh_white, s_thresh_white), get_hls_thresh(img, h_thresh_white_2, s_thresh_white_2), do_or=True)
        # hls_thresh = alt_combine(alt_combine(get_hls_thresh(img, h_thresh_white_2, s_thresh_white_2), get_hls_thresh(img, s_thresh=s_thresh_white)),
        #                          get_hls_thresh(img, h_thresh_white))

        # hls_thresh = alt_combine(get_hls_thresh(img, h_thresh_white_2, s_thresh_white_2), get_hls_thresh(img, h_thresh_white, s_thresh_white), do_or=True)
        # hls_thresh = alt_combine(get_hls_thresh(img, h_thresh_white, s_thresh_white), get_hls_thresh(img, h_thresh, s_thresh) , do_or=True)
        ans = alt_combine(ans, hls_thresh)
        # ans = hls_thresh
        # , get_hls_thresh(img, h_thresh, s_thresh)

        # test HLS


        # overlaying detection with parent image

        # warp_zero = np.zeros_like(ans).astype(np.uint8)
        # color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
        # color_warp[(ans == 1), :1] = 255

        # # cv2.fillPoly(color_warp, np.int_([ans.reshape(-1)]), (0, 255, 0))

        # ori_images[count] = cv2.addWeighted(ori_images[count], 1, color_warp, 0.3, 0)

        # # ori_images[count][(ans == 1)] =
        # count += 1

        ans = pers_transform(ans)
        # TODO remove mask for tough curves
        # ans = mask(ans)

        imgs.append(ans)

        #plot = plt.imshow(ans, cmap="gray")
    di(imgs)
#    plt.show()



# di(imgs)

# plt.imshow(ori_images[6])
# plt.show()

# im = mpimg.imread(images[5])

# im = pers_transform(im)

# R = im[:, :, 0]

# plt.imshow(R, cmap='gray')
# plt.show()

#di(imgs)

from pipeline import process, display_lines, draw_lines

# img = imgs[0]
# binary_warped = pers_transform(img)
# plt.imshow(binary_warped, cmap='gray')
# plt.show()

# displaying detected lanes

# (out_img, left_fitx, right_fitx, ploty)


# for i in range(len(imgs)):
#     (left_fit, right_fit) = process(imgs[i])
#     ori_images[i] = draw_lines(imgs[i], ori_images[i], left_fit, right_fit, pers_transform)

# di(ori_images)

# plt.tight_layout()
# plt.show()


# plt.imshow(result)

# [display_lines(*process(img)) for img in imgs]



from moviepy.editor import VideoFileClip
from IPython.display import HTML
from collections import deque

# Converts a clip from raw file to a file with lane overlay
def write_clip(input_file, output_file, function):
    clip = VideoFileClip(input_file)

    global left_line_points, right_line_points, fail_count, curvature_queue
    left_line_points = None
    right_line_points = None
    fail_count = 0
    curvature_queue = deque([])

    white_clip = clip.fl_image(function) #NOTE: this function expects color images!
    white_clip.write_videofile(output_file, audio=False)


# global
count = 0


def process_frame(image, debug=True):


    copy = np.copy(image)
    copy = undistorter(copy)

    gray = cv2.cvtColor(copy, cv2.COLOR_RGB2GRAY)

    abs_args = (gray, 35, 120, 15, 40)  # (img, 35, 120, 30, 40)
    mag_args = (gray, 10, 255, 9)  # (img, 10, 200, 9)
    grad_args = (gray, 0.7, 1.3, 9)

    h_thresh = (0, 79)
    s_thresh = (130, 255)

    h_thresh_white = (100, 130)  # (100, 140) # (110, 150) (15, 90)(0, 90)
    s_thresh_white = (0, 15)  # (0, 20) # (0, 30) (90, 255)

    h_thresh_white_2 = (0, 90)  # (0, 30)
    s_thresh_white_2 = (90, 255)
    # TODO OR combination between magnitude/gradient and x/y sobels
    ans = alt_combine(abs_thresh(*abs_args), alt_combine(mag_thresh(*mag_args), grad_thresh(*grad_args)), do_or=True)

    hls_thresh = get_hls_thresh(copy, h_thresh_white_2, s_thresh_white_2)
    ans = alt_combine(ans, hls_thresh)


    # TODO remove mask for tough curves
    # ans = mask(ans)

    if debug:
        global count
        count += 1
        ans = pers_transform(ans)
        # image[(ans == 1), :] = 0
        # image = np.dstack((ans, ans, ans))
        # image[(ans == 1), :] = 255
        if count >= 40:
            plt.imshow(ans)
            plt.show()
            import ipdb
            ipdb.set_trace()
    else:
        ans = pers_transform(ans)
        (left_fit, right_fit) = process(ans)
        image = draw_lines(ans, image, left_fit, right_fit, pers_transform)

    return image


def test():
    vinp = 'project_video.mp4'
    vout = 'project_video_solution.mp4'

    vinp = "project_video_error_short.mp4"
    vout = "project_video_error_solution_test.mp4"

    write_clip(vinp, vout, process_frame)

    # # plt.imshow(pers_transform(img))
    # # plt.show()
    # images = list(map(lambda x: mpimg.imread(x), glob.glob("test_images/*.jpg")))
    # tar = []
    # for i in images:
    #     #tar.append(i)
    #     tar.append(pers_transform(i))
    #     tar.append(pers_transform(pers_transform(i), reverse=True))
    # tar = tar[10:]
    # ###
    # # im = plt.imshow(tar[0], interpolation='nearest')
    # # fig = plt.gcf()
    # # ax = plt.gca()

    # # class EventHandler:
    # #     def __init__(self):
    # #         fig.canvas.mpl_connect('button_press_event', self.onpress)

    # #     def onpress(self, event):
    # #         if event.inaxes != ax:
    # #             return
    # #         xi, yi = (int(round(n)) for n in (event.xdata, event.ydata))
    # #         value = im.get_array()[xi, yi]
    # #         color = im.cmap(im.norm(value))
    # #         print (xi, yi, value, color)

    # # handler = EventHandler()
    # # plt.show()


    # from visualize import draw_images as di

    # di(tar)


if __name__ == "__main__":
    # pass
    test_images()

