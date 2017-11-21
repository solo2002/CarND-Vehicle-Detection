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

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

All the code for this project is in the file of vehicle_detection.ipynb.

### Histogram of Oriented Gradients (HOG)

#### 1. How to extract HOG features from the training images.

The code for this step is contained in the code cell 6 of the IPython notebook.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is some example of the `vehicle` and `non-vehicle` classes:

![alt text](https://github.com/solo2002/CarND-Vehicle-Detection/blob/master/output_images/dataset_visual.jpg?raw=true)

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`) in code cell 7.  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text](https://github.com/solo2002/CarND-Vehicle-Detection/blob/master/output_images/hog_example.jpg?raw=true)

#### 2. How to determine my final choice of HOG parameters.

I tried various combinations of parameters in code cell 8 and 9 through visualize the results of HOG features as well as the performance of trained classifier. Here are some example output of different parameters settings.

![alt text](https://github.com/solo2002/CarND-Vehicle-Detection/blob/master/output_images/tuning_windows_setting1.jpg?raw=true)

![alt text](https://github.com/solo2002/CarND-Vehicle-Detection/blob/master/output_images/tuning_windows_setting0.jpg?raw=true)

For the color space, 'YCrCb' give the best result in my case. Meanwhile, time for extracting features is also another important factor to consider. Eventually, I select following values:

cspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb

orient = 9

pix_per_cell = 8

cell_per_block = 2

hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"

spatial_size = (32, 32)

hist_bins = 32

spatial_feature = True

hist_feature = True

hog_feature = True

#### 3. How to train the classifier 

In Cell 9, I trained a linear SVM using the normalized HOG features. The features are first normalized as follows:

X = np.vstack((car_features, noncar_features)).astype(np.float64)  

# Fit a per-column scaler

X_scaler = StandardScaler().fit(X)

# Apply the scaler to X

scaled_X = X_scaler.transform(X)

# Define the labels vector

y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))#3*len(noncar_features)

Then, the overall dataset is split as training and test data:

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.1, random_state=rand_state)

Here, the split ratio is 9 : 1 in stead of 8 : 2, since the number of data is limited.

Here is the output of test images:
![alt text](https://github.com/solo2002/CarND-Vehicle-Detection/blob/master/output_images/find_car_example.jpg?raw=true)

### Sliding Window Search

#### 1. How to implement a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

