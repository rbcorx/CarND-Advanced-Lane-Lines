import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

# 4:44

# binary filter methods


def norm(s):
    return 255 * np.absolute(s) / np.max(s)


def sobel(img, x=0, norm=True, **kwargs):
    s = np.absolute(cv2.Sobel(img, cv2.CV_64F, x, abs(x - 1), **kwargs))
    if norm:
        s = 255 * np.absolute(s) / np.max(s)
        # print(s[:30])
    return s


def thresh(s, mi, mx, zeros=None):
    if zeros is None:
        zeros = np.zeros_like(s)
    zeros[(s >= mi) & (s <= mx)] = 1
    return zeros


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


def get_hls_thresh(img, h_thresh, s_thresh):

    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    H = hls[:, :, 0]
    S = hls[:, :, 2]
    # mask_H = thresh(H, h_thresh[0], h_thresh[1])
    #mask_S = thresh(S, s_thresh[0], s_thresh[1])
    thresh = (15, 100)
    binary = np.zeros_like(H)
    binary[(H > h_thresh[0]) & (H <= h_thresh[1]) & (S > s_thresh[0]) & (S <= s_thresh[1])] = 1
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
        dst = np.float32([[xl, 0], [xr, 0], [xr, 677], [xl, 677]])
        p_matrix = cv2.getPerspectiveTransform(src, dst)
        p_inv_matrix = cv2.getPerspectiveTransform(dst, src)
        rendered = True
    m = p_matrix if (not reverse) else p_inv_matrix
    return cv2.warpPerspective(image, m, (image.shape[:2][::-1]), flags=cv2.INTER_LINEAR)

# perspective transform ends


# testing code

images = glob.glob("test_images/*.jpg")
imgs = []
for fimg in images:

    img = mpimg.imread(fimg)
    #img = pers_transform(img)

    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    abs_args = (img, 35, 120, 30, 40)
    mag_args = (img, 10, 200)
    grad_args = (img, 0.7, 1.3)
    h_thresh = (0, 79)
    s_thresh = (100, 255)
    # TODO OR combination between magnitude/gradient and x/y sobels
    ans = combine(abs_thresh(*abs_args), mag_thresh(*mag_args), grad_thresh(*grad_args))
    # , get_hls_thresh(img, h_thresh, s_thresh)

    # test HLS
    #ans =
    imgs.append(ans)

    #plot = plt.imshow(ans, cmap="gray")

#    plt.show()

from visualize import draw_images as di

di(imgs)


def test():
    pass
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
    test()

