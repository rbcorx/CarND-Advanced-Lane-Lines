import numpy as np
import cv2
import matplotlib.pyplot as plt

# globals
left_fit = None
right_fit = None


navg = 10
left_prev = []
right_prev = []


def averager(left_fit, right_fit, add=None):
    global left_prev, right_prev
    if len(left_prev) >= navg:
        left_prev = left_prev[1:]
        right_prev = right_prev[1:]

    if add is None:
        left_prev.append(left_fit)
        right_prev.append(right_fit)
    else:
        left_prev.append(add[0])
        right_prev.append(add[1])

    return (np.mean(left_prev, axis=0), np.mean(right_prev, axis=0))


def sanity_checks(left_fit, right_fit, recover=False, use_avg=False):
    # calculated values make sese or do blind search
    # copy previous values if even blind search doesn't yeild satisfactory results

    # constant value makes sense, in the ballpark of distance estimated previously

    # constant values don't drastically change at once

    alpha = (30, 30, 0.5)  # 3000%, 3000%, 50% tolerence to change

    is_sane = True
    # print (len(left_prev))
    if len(left_prev) == navg:


        # taking last elem for checks
        mean_l = left_prev[-1]
        mean_r = right_prev[-1]

        if use_avg:
            # taking mean instead of last elem
            mean_l = np.mean(left_prev, axis=0)
            mean_r = np.mean(right_prev, axis=0)

        for mean, fit in ((mean_l, left_fit), (mean_r, right_fit)):
            # print("sanity checks GO! ::")
            # print (mean)
            # print (fit)
            # print (list(map(lambda x: abs(x[0] - x[1]) / abs(x[0]) < x[2], zip(mean, fit, alpha))))
            is_sane = is_sane and all(list(map(lambda x: abs(x[0] - x[1]) / abs(x[0]) < x[2], zip(mean, fit, alpha))))

    if recover and (not is_sane):
        return (mean_l, mean_r)

    return is_sane


def radius_and_dist(warped, left_fit, right_fit):
    """calculates radius of curvature and distance of car from center"""
    norm = 1000

    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])

    leftx = np.array([(y ** 2) * left_fit[0] + y * left_fit[1] + left_fit[2]
                      for y in ploty])
    rightx = np.array([(y ** 2) * right_fit[0] + y * right_fit[1] + right_fit[2]
                      for y in ploty])

    y_eval = np.max(ploty)
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30 / 720  # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

    # Fit new polynomials to x, y in world space
    left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx *
                             xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval *
                           ym_per_pix + left_fit_cr[1]) ** 2) **
                     1.5) / np.absolute(2 * left_fit_cr[0])
    right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix +
                            right_fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit_cr[0])
    # Now our radius of curvature is in meters

    # radius = [left_curverad, right_curverad]
    # print(left_curverad, 'm', right_curverad, 'm')

    lx = ((y_eval * ym_per_pix / 2) ** 2) * left_fit_cr[0] + (y_eval * ym_per_pix / 2) * left_fit_cr[1] + left_fit_cr[2]
    rx = ((y_eval * ym_per_pix / 2) ** 2) * right_fit_cr[0] + (y_eval * ym_per_pix / 2) * right_fit_cr[1] + right_fit_cr[2]

    center_mid_x = (lx + rx) / 2 / xm_per_pix

    img_center_x, img_center_y = warped.shape[1] / 2, warped.shape[0] / 2

    dist = np.sqrt(((img_center_y - y_eval / 2) * ym_per_pix) ** 2 +
                   ((img_center_x - center_mid_x) * xm_per_pix) ** 2)

    # print ("distance from center : {}".format(dist))

    radius = left_curverad if abs(norm - left_curverad) < abs(norm - right_curverad) else right_curverad
    # print (radius, "m : radius found")

    return (radius, dist)


def blind_lane_search(binary_warped):
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

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

    # visualization

    # Generate x and y values for plotting


    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
    # import ipdb
    # ipdb.set_trace()
    return (left_fit, right_fit)


def targeted_lane_search(left_fit, right_fit, binary_warped):
    # Assume you now have a new warped binary image
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!

    plt.imshow(binary_warped)

    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) +
                       left_fit[1] * nonzeroy + left_fit[2] - margin)) &
                      (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                       left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # import ipdb
    # ipdb.set_trace()
    # print("debug outputLLL@@@@")
    # print(lefty.shape)
    # print(leftx.shape)

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return (left_fit, right_fit)


def process(binary_warped):
    global left_fit, right_fit
    add_history = None
    sanity_retained = True
    if left_fit is not None:
        try:
            (left_fit_new, right_fit_new) = targeted_lane_search(left_fit, right_fit, binary_warped)
            if not sanity_checks(left_fit_new, right_fit_new):
                # print("this is insane TARGETED!!!!! $$##")
                sanity_retained = False
                # TODO BETA
                # (left_fit_new, right_fit_new) = blind_lane_search(binary_warped)

            # TODO handle errors more gracefully
        except:
            sanity_retained = False
            # print("this is insane EXCEPTION!!!!! $$##")
            # import ipdb
            # ipdb.set_trace()
    else:
        sanity_retained = False

    if not sanity_retained:
        sanity_retained = True
        try:
            (left_fit_new, right_fit_new) = blind_lane_search(binary_warped)
        except:
            (left_fit_new, right_fit_new) = (left_fit, right_fit)

        res = sanity_checks(left_fit_new, right_fit_new, recover=True)
        if res is not True:
            sanity_retained = False
            # add_history = (left_fit_new, right_fit_new)
            # print("this is insane BLIND!!!!! $$##")
            (left_fit_new, right_fit_new) = res
    # import ipdb
    # ipdb.set_trace()
    (left_fit_new, right_fit_new) = averager(left_fit_new, right_fit_new, add_history)

    left_fit = left_fit_new
    right_fit = right_fit_new

    measures = radius_and_dist(binary_warped, left_fit, right_fit)

    return (left_fit, right_fit, measures)


def draw_lines(warped, image, left_fit, right_fit, pers_transform):
    ploty = np.linspace(0, warped.shape[0] - 1, warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    pts = pts.reshape(pts.shape[1], 2)

    # Draw the lane onto the warped blank image
    # import ipdb
    # ipdb.set_trace()
    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))


    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = pers_transform(color_warp, True)
    # Combine the result with the original image
    return cv2.addWeighted(image, 1, newwarp, 0.3, 0)


def put_text(img, measures):
    text = "ROC: {:.3f}, distance: {:.3f}".format(*measures)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, (10, 50), font, 1, (255, 255, 255), 2)


# globals
fig = None
count = 0


def display_lines(out_img, left_fitx, right_fitx, ploty):
    global fig, count

    if not fig:
        fig = plt.figure(figsize=(17, 7))

    a = fig.add_subplot(8 // 3 + 1, 3, (count + 1))  # 3
    count += 1

    plt.imshow(out_img)
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    plt.xlim(0, 1280)
    plt.ylim(720, 0)
    a.set_title(str(1) + "th image")




