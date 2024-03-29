## Writeup
---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.
You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first and third code cell of the IPython notebook `Vehicle_Detection.ipynb`.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text](./output_images/car.png)
![alt text](./output_images/non-car.png)


I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text](./output_images/hog.JPG)

#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and `orient = 12` , `pix_per_cell = 8` , `cell_per_block = 2` and `YCrCb` color space worked pretty well, I also combined them with color features, I set `spatial_size = (16, 16)` and `hist_bins = 128` for them.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HOG features and color features(bin spatial and color histogram). The code for this step is contained in the third code cell of the IPython notebook `Vehicle_Detection.ipynb`.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search window positions at 4 scales in the bottom(380 < y < 680) of the image and came up with this :

![alt text](./output_images/sliding_window.jpg)

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on 4 scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text](./output_images/result1.jpg)
![alt text](./output_images/result2.jpg)
![alt text](./output_images/result3.jpg)
![alt text](./output_images/result4.jpg)

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://github.com/hankkkwu/SDCND-Vehicle_Detection/blob/master/combined_result.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text](./output_images/frame1.JPG)
![alt text](./output_images/frame2.JPG)
![alt text](./output_images/frame3.JPG)

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all three frames:
![alt text](./output_images/label_map.jpg)

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text](./output_images/final_output.jpg)


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, First I took features from `vehicle` and `non-vehicle` images using HOG, bin spatial and color histogram techniques, then I used those features to train a linear SVM classifier. Second, I apply Sliding Window Search technique to search for vehicles in a image, then draw the bounding boxes on the vehicles it detected. I think this pipeline might fail when there are others objects(like motocrycles, bikes, or pedestrian etc.) near by vehicles, because my SVM model can only classify car and non-car. It might also fail to distinguish each vehicle with individual bounding box when there are more vehicles on the road, I might improve it by using YOLO technique, if I am going to pursue this project further.  
