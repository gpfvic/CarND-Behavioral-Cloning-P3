# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model.png "Model architecture"
[image2]: ./examples/center_lane.jpg "Center of the lane"
[image3]: ./examples/left_2021_07_26_08_03_43_250.jpg "Left of the lane"
[image4]: ./examples/right_2021_07_26_08_03_43_250.jpg "Right of the lane"
[image5]: ./examples/left_flip.jpg "Origin image with left side"
[image6]: ./examples/right_flip.jpg "Fliped image"
[image7]: ./examples/visualize_loss.png "validation loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is designed according to the Nvidia's paper. My model consists of several convolution layers and fully connected layers(model.py lines 71-90).

The model includes RELU layers to introduce nonlinearity (model.py lines 74-82), and the data is normalized in the model using a Keras lambda layer (model.py line 73). 

The model architecture is shown follows:
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
cropping2d_1 (Cropping2D)    (None, 80, 320, 3)        0
_________________________________________________________________
lambda_1 (Lambda)            (None, 80, 320, 3)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 38, 158, 24)       1824
_________________________________________________________________
spatial_dropout2d_1 (Spatial (None, 38, 158, 24)       0
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 17, 77, 36)        21636
_________________________________________________________________
spatial_dropout2d_2 (Spatial (None, 17, 77, 36)        0
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 7, 37, 48)         43248
_________________________________________________________________
spatial_dropout2d_3 (Spatial (None, 7, 37, 48)         0
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 5, 35, 64)         27712
_________________________________________________________________
spatial_dropout2d_4 (Spatial (None, 5, 35, 64)         0
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 3, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 6336)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               633700
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dropout_2 (Dropout)          (None, 50)                0
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 770,619
Trainable params: 770,619
Non-trainable params: 0
```

#### 2. Attempts to reduce overfitting in the model

The model contains several dropout layers in order to reduce overfitting (model.py lines 21). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 107).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I driving for nearly 7 lays and 5 laps in a counter-clockwise direction. The data size is around 1Gb.

I used the three cameras'images and angles as training data and the angles are corrected by 20 degrees. Then I filp the images and negtive the angles to augmentate the training dataset.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to try to use LeNet model with a lamdba layer for normalization and a cropping layer to crop input image to appropriate size mentioned in the lecture notes. The car will easily drift out of the center of lane. It also works bad when the roadside is empty and so on. 

Then I use the Nvidia's model architecture. This time the problem is the overfitting as the training loss decreases linearly while the validation loss fluctuates up and down. So I tried to add some dropout layers to reduce the overfitting problem. 

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. Also I collect more data for training.

Finally the model is well trained and the the car can keep on the center of road all the time without leaving the road.

#### 2. Final Model Architecture

Here is a visualization of the architecture.

![The final model architecture][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 7 laps on track one using center lane driving. Here is an example image of center lane driving:

![Center of the lane][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to predict the steering angles.
![left of the lane][image3]
![right of the lane][image4]

Then I repeated this process on the counter-clockwise direction and collect 6 laps.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![Origin image on left side][image5]
![flipped image on right side][image6]


After the collection process, I had 25822 number of origin data (image, angle) and the same number of flipped data.


I finally randomly shuffled the data set and put 80% of the data into a validation set. All the data was shuffled randomly. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. I set the epoch number as 10. I used an adam optimizer so that manually training the learning rate wasn't necessary. The training and validation loss is shown in the following image. The validation is still little fluctuation but the trend looks like reasonable.  

![training and validation loss][image7]


