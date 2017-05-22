## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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


Note: Instead of using OOP, I have mainly implemented a functional programming architecture augmented by globals where neccessary for the sake of rapid prototyping.

IMP files:
* test.py binary filters and frame processing functions
* pipeline.py pipeline for finding and drawing lane lines
* undistort.py undistortion functions
* test.ipynb for visulization only
* project_video_solution.mp4 output video


[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video_solution.mp4 "Video"
[binary]: ./examples/binary.png "Binary 1"
[binary_ori]: ./examples/binary_ori.png "Binary 2"
[binary_warp]: ./examples/binary_warp.png "Binary 3"
[corrected]: ./examples/corrected.png "corrected"
[final]: ./examples/final.png "final"
[ori]: ./examples/ori.png "original image"
[pers_ori]: ./examples/pers_ori.png "perspective original"
[pers_warp]: ./examples/pers_warp.png "perspective warped"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the file `undistort.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

![alt text][corrected]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
![alt text][ori]
![alt text][corrected]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

In the file `test.py`, starting from line no. 19, I have created filter methods which apply thresholds for:
* absolute sobel gradients in x and y
* magnitude of sobel gradients
* gradient of sobelx and sobely
* R channel
* H and S channel for HLS color space

A binary filtering pipeline is used at line 220 in `test.py` in the function `process_frame` which does the following in order:
* reads image in RGB
* converts a copy to grayscale for sobel operations
* creates a binary using thresholds settings using the grayscale image with:
    * absolute sobel gradients in x and y
    * magnitude of sobel gradients
    * gradient of sobelx and sobely
* creates a binary using thresholds for H and S after coversion to HLS colorspace using the original image
* combines the bianaries
* performs perspective transformation
* finds lane lines and measurements

I used a combination of color and gradient thresholds to generate a binary image. Here's an example of my output for this step.  (note: this is not actually from one of the test images)

I discovered while experimentation that a more robust white lane detection can be done using multiple threshold windows for H and S channels but it also introduces noise. I have demonstrated it in the function `test_images` but haven't used it for processing frames.
This is useful in the challenge videos to detect difficult lane lines but I have skipped that for now for the sake of brevity.

![alt text][binary_ori]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `pers_transform()`, which appears in lines 129 through 142 in the file `test.py` The function takes as inputs an image (`image`), and (`reverse`), which indicates if the image has to warped back to normal using the inverse matrix.  I chose the hardcode the source and destination points in the following manner by manual selection:

```python
src = np.float32([[561, 474], [725, 474], [1040, 677], [254, 677]])
xr = (1040 + 725)//2
xl = (254 + 561)//2
dst = np.float32([[xl, 0], [xr, 0], [xr, 720], [xl, 720]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][pers_ori]
![alt text][pers_warp]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

In the file `pipeline.py`, I have a function `process` which:
* takes in a warped binary to be used to find the lane lines
* selects strategy to find lane lines (blind, from scrach or targeted search using previously found lane lines)
    * the functions `blind_lane_search` and `targeted_lane_search` are responsible for detecting lane lines and fitting polynomials using the sliding window technique from scratch and by reusing previous detections respectively
* performs sanity checks which are basically checks for deviation of fit coefficients from previously found values to ascertain high confidence detection within defined deviation tolerance limits
This results in a remarkable boost in stability and robustness of lane lines detected as we get to know when we don't have a good result and that way we can switch blind lane searching or reuse previous detections
* performs averaging over last N frames to reduce jitter and increase stability
* extracts radius of curvature and distance of car from center

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 69 through 115 in my code in `pipeline.py` within the function `radius_and_dist`

I use the coefficients of the left and right lane fits calculated thus far to find ROC as follows:
* first create x, y points for both lanes
* convert the points to real world space using the coefficients of metres_per_pixel in x and y
* refit the points in a polynomial to find the real world fit coefficients
* use the coefficents in the formulae ((1 + (2Ay + B)**2) ** 3/2) / |2A| to find the ROC

I only print out the value which is closer to the expected value of 1000m between the left and right lane ROCs

I calculate the distance of car from the center by finding the lane center and calculating it's distance from the center of the image.

The lane center can be found by finding the x coordinate for left and right lanes corresponding to the y center of the image converted to real world value system as demonstrated in the function.

I then convert the lane center back to the image coordinate system and find it's distance form the image center, and finaly convert this value into metres.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 233 through 271 in my code in `pipeline.py` in the function `draw_lines()`.  Here is an example of my result on a test image:

![alt text][final]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video_solution.mp4)
![alt text][video1]
(project_video_solution.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

The pipeline fails at different brightness levels as expected because the gradient filters fail spectaculary in that context and have to be changed/readjusted.

The pipeline also fails when the lane lines disappear on the road. Though I have implemented a recovery solution for that where I reuse the previously found coefficient values, it's not a perfect solution.

I could have used an outlier rejection filter to make the lines more robust but decided against it due to time constraints. I intend to implement it as an exercise soon though.

The current H and S filters almost fail to find white lines properly in varied brightness conditions. This is due to the fact that the white lines lie in two regions in the H and S channels: HIGH H, LOW S and LOW H, HIGH S, something like:
```
h_thresh_white_high (100, 130)
s_thresh_white_low = (0, 15)

h_thresh_white_low = (0, 90)
s_thresh_white_high = (90, 255)
```

I would also like to add multiple H and S filter windows to better detect the white lane lines in different brightness conditions.

