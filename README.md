# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./examples/Nvidia.png "Nvidia Neural Net Acrhitecture"

Overview
---
This repository contains my implementation for Udacity SDCND project "Behavioral Cloning". The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior. Udacity has provided the image data of the car running the simulator.
* Design, train and validate a convolutional neural network model that predicts a steering angle from image data.
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.

In the following sections, I will describe how each of the Rubric points were met.

### Files submitted

* model.py : This file contains the neural network implementation.
* drive.py : This file contains the code to drive the car autonomously in the simulator based on the model.
* model.h5 : model file that contains the learned weights of the neural network.
* video.mp4 : Video of the car running in track1 of the simulator in autonomous mode.

### Quality of Code

#### Is the code functional?
I am able to drive the car autonomously in the Udacity simulator using the trained model successfully. video.mp4 contains a video of the car running in the simultaor autonomously

#### Is the code usable and readable?
prepare_data.py file has the python code to unzip the Udacity provided data.zip file and copy the image files to /opt/ directory. model.py contains the code for data generator, NVIDIA neural network to train and save the model.

Code is organized in to functions and comments are added to code.


### Model Architecture and Training Strategy

#### Architecture

I have used the NVIDIA convolutional neural network that was suggested in the Udacity project instructions. Details of the architecture can be found [here.](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) The network has about 27 million connections and 250 thousand parameters.

![alt text][image1]

Input size of the above neural net is 66x200x3. The original images produced by our car camera is 160x320x3. During the initial stages of neural net, the upper and some lower part of the image was removed. This is accomplished by the following lines of code.

```python
# Crop the image to remove irrelevant part of the image
model.add(Cropping2D(cropping=((70,25),(0,0))))  
```

This makes the size of the images to 65x320x3. I have decided use this input size to neural network.

I tried to generate my own training data by running the car in track-1 of the Udacity simulator. I found that controlling the car in the track and generating the data quite challenging. I found that the data provided by Udacity [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) is reasonably good. So decided to try to train the network using the Udacity provided data and then generate my own data only if it is required later. 

#### Data Generator
A `generator()` function was implemented in `model.py` to return the data in batches to the model during training to reduce the memory usage. Car in simulator has three cameras: center, left and right. Each record in the csv file contains these three camera snapshots and the corresponding steering angle. Since there is only one steering angle associated with three images, I have decided to apply a correction factor of 0.2. For left images, steering angle is increased by 0.2 and for right images, steering angle is decreased by 0.2.

#### Data Augmentation
The data provided by Udacity is captured by running the car in Simulator Test mode and mostly turns towards left. So the model will have the bias to turn left. To overcome this limitation, augmented data is generated by flipping the data horizontally as suggested in the lessons. This is implemented in `generator()` in `model.py`. Image flipping is implemented using the OpenCV function `cv2.flip()`. Steering angle is multiplied by -1 for the flipped images.

Another technique that was suggested in the leasson to avoid left turn bias was to drive car in opposite direction in test track, but I haven't done it.



#### Has an appropriate model architecture been employed for the task?

#### Has an attempt been made to reduce overfitting of the model?

#### Have the model parameters been tuned appropriately?

##### Is the training data chosen appropriately?


### Architecture and Training Documentation

#### 



This repository contains starting files for the Behavioral Cloning Project.

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

We have provided a simulator where you can steer a car around a track for data collection. You'll use image data and steering angles to train a neural network and then use this model to drive the car autonomously around the track.

We also want you to create a detailed writeup of the project. Check out the [writeup template](https://github.com/udacity/CarND-Behavioral-Cloning-P3/blob/master/writeup_template.md) for this project and use it as a starting point for creating your own writeup. The writeup can be either a markdown file or a pdf document.

To meet specifications, the project will require submitting five files: 
* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

Creating a Great Writeup
---
A great writeup should include the [rubric points](https://review.udacity.com/#!/rubrics/432/view) as well as your description of how you addressed each point.  You should include a detailed description of the code used (with line-number references and code snippets where necessary), and links to other supporting documents or external references.  You should include images in your writeup to demonstrate how your code works with examples.  

All that said, please be concise!  We're not looking for you to write a book here, just a brief description of how you passed each rubric point, and references to the relevant code :). 

You're not required to use markdown for your writeup.  If you use another method please just submit a pdf of your writeup.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

The following resources can be found in this github repository:
* drive.py
* video.py
* writeup_template.md

The simulator can be downloaded from the classroom. In the classroom, we have also provided sample data that you can optionally use to help train your model.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires you have saved the trained model as an h5 file, i.e. `model.h5`. See the [Keras documentation](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) for how to create this file using the following command:
```sh
model.save(filepath)
```

Once the model has been saved, it can be used with drive.py using this command:

```sh
python drive.py model.h5
```

The above command will load the trained model and use the model to make predictions on individual images in real-time and send the predicted angle back to the server via a websocket connection.

Note: There is known local system's setting issue with replacing "," with "." when using drive.py. When this happens it can make predicted steering values clipped to max/min values. If this occurs, a known fix for this is to add "export LANG=en_US.utf8" to the bashrc file.

#### Saving a video of the autonomous agent

```sh
python drive.py model.h5 run1
```

The fourth argument, `run1`, is the directory in which to save the images seen by the agent. If the directory already exists, it'll be overwritten.

```sh
ls run1

[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_424.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_451.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_477.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_528.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_573.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_618.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_697.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_723.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_749.jpg
[2017-01-09 16:10:23 EST]  12KiB 2017_01_09_21_10_23_817.jpg
...
```

The image file name is a timestamp of when the image was seen. This information is used by `video.py` to create a chronological video of the agent driving.

### `video.py`

```sh
python video.py run1
```

Creates a video based on images found in the `run1` directory. The name of the video will be the name of the directory followed by `'.mp4'`, so, in this case the video will be `run1.mp4`.

Optionally, one can specify the FPS (frames per second) of the video:

```sh
python video.py run1 --fps 48
```

Will run the video at 48 FPS. The default FPS is 60.

#### Why create a video

1. It's been noted the simulator might perform differently based on the hardware. So if your model drives succesfully on your machine it might not on another machine (your reviewer). Saving a video is a solid backup in case this happens.
2. You could slightly alter the code in `drive.py` and/or `video.py` to create a video of what your model sees after the image is processed (may be helpful for debugging).

### Tips
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

## How to write a README
A well written README file can enhance your project and portfolio.  Develop your abilities to create professional README files by completing [this free course](https://www.udacity.com/course/writing-readmes--ud777).

