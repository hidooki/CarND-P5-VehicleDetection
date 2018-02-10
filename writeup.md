
# **Vehicle Detection**

The goals of this project is to write a pipeline that detects other vehicles on the road from a video stream. The steps are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

It is graded according to the following  [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.jpg
[image2]: ./output_images/A_car_hog_visualisation.jpg
[image22]: ./output_images/Not_a_car_hog_visualisation.jpg

[image3]: ./output_images/search_window.jpg
[image4]: ./output_images/boxes.jpg
[image5]: ./output_images/boxes_and_heat.jpg

[image7]: ./output_images/cars_found.jpg
[video1]: ./project_video_result.mp4



---
### Submitted Files

The following files are included with the project:

* project writeup
* code: `runner.py`, `classifier.py`, `features.py`, `utils.py`
* images: included in the `output_images` folder
* optimal model parameters pickled in the `params` folder
* video result: `project_video_result.mp4`



### Histogram of Oriented Gradients (HOG)

#### HOG features extraction

Feature extraction is done in the `features.py` module. It contains the lesson functions for extracting basic features (spatial information, color histogram and histograms of oriented gradients), standalone and also wrapped into a single `feature_extractor()` function.

The feature extraction code uses the `hog()` function from the `skimage.feature` package, which was wrapped in the `get_hog_features()` lesson function.

I used a preprocessing step before extracting HOG features, which consisted in converting the image to a different color space. I experimented with different color spaces, suspecting that the choices like HLS or YCrCb might work better as they separate the luminosity from the essential feature-related information.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example of a **car** using the `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

and here is an example of a **non-car** using the same parameters:

![alt text][image22]

#### Final choice of HOG parameters

I found that the choice of color space was quite important, with more 'traditional' choices as RGB not performing as well as color spaces that separate the luminosity from the color features. In particular, the YCrCb space seemed to perform well when classifying the image set provided.
I experimented with different values of the `orient` parameter, trying to find the right balance between precision and extraction time. I found that, when paired with color histogram and spatial features, a choice of orient=8 works well. For the other HOG parameters I used the values suggested in the lesson, given that later at classification time I used a scaling parameter to zoom in/out of the image.

In ended up with YCrCb for the color space, orient = 8, 8 pixels per cell and 2 cells per HOG block. I ran the HOG feature extraction on all three color channels.

#### Training the classifier

I trained a linear SVM classifier using the sklearn library. In addition to the HOG features, I also used spatial binning where I resized the images into 32 x 32 blocks. As a final set of features I used a color histogram, where I grouped each color channel into 32 bins. The set of all these features put together ()

In training the classifier I used the full set of images provided (8,792 cars and 8,968 non-cars), where I kept 20% for test data and I used the other 80% for training. When using all the features described above (7,872 features per image in total), I got a 99.38% accuracy on the test set.

The code is contained in the `classifer.py` module.


### Sliding Window Search

#### Sliding window search

I implemented a sliding window search in the `runner.py` module. As suggested in the lesson, I wrote a function `find_cars_multiple_scales()` that performs systematic window searches for multiple window sizes (controlled by the 'scale' parameter).

I first restricted the search area to the bottom half of each image, where cars are (usually) to be found.


![alt text][image3]



I tried a few values of the scale parameter and noticed that, as far the area of the picture I as focusing on (above), scales lower than 1 introduced a lot of false positives while scales bigger than 3 produced not cars. I then experimented with layering rows on increasing window sizes, while running the search over a short clip (between second 9 and second 11) from the video and re-iterating. Instead of using an 'image overlap percentage', I used a 'cell per step' parameter to control the density of the search window grid. I settled for a small step equal to 2 cells, which seemed to provide enough granularity given that the search window had 8 cells (64 pixels).


#### Examples of pipeline result images

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are the results of the cars search on the six test images provided with the project:

![alt text][image4]

To reduce the number of false negatives, I first made sure that the scale is high enough (a scale lower than 1 will produce many false negatives by picking up plenty of garbage in the mid section of the image).

Second, when implementing the pipeline I maintained a list of trailing frames, whose heatmaps I then integrated together before thresholding at a higher level. As false negative tend to come in fleeting frames, they will not clear the threshold bar in the heatmap sum of nearby frames and so will get filtered out.

### Video Implementation

#### Video output

Here's a [link to my video result](./project_video_result.mp4)


#### Pipeline description

The pipeline is implemented in the `make_pipeline()` function in the `runner.py` module, which is really a pipeline factory. For each image, I ran the sliding window search at multiple scales (as described above). I aggregate all positive detections into a heatmap. In order to filter out false positives, I maintain a list of heatmaps from the prior frames. (The size of the list is constant, a parameter into the pipeline factory). I add up all these heatmaps and threshold the sum at a high enough level. I used a threshold of 5 for every sum of 10 past images, which seemed to filter out the vast majority of false positives.  

I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap sum.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from the series six test images provided with the project, and the bounding boxes then overlaid on the last frame of video:


![alt text][image5]

After applying `scipy.ndimage.measurements.label()` on the heatmap in the last frame in the series above, and overlaying the resulting bounding boxes on the original image, we get the image below.  Notice that after applying the threshold, the false positive disappeared.

![alt text][image7]



---

### Discussion


I have to say that I find this technique of object detection a bit flakey. The way these image features organize themselves into cars and not-cars by a SVM classifier seems like an accident waiting to happen :-). I would want to train the classifier on perhaps **billions** of images after driving around the Earth, in order to make sure that really it has "seen anything under the sun", or else there is always a chance of hitting some "invisible" (i.e. never seen before) object.
