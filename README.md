# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)
[image1]: ./examples/Nvidia.png "Nvidia Neural Net Acrhitecture"
[image2]: ./examples/Final_architecture.png "Final Acrhitecture"
[image3]: ./examples/center.jpg "Center camera image"

Overview
---
This repository contains my implementation for Udacity SDCND project "Behavioral Cloning". The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior. Udacity has provided the image data of the car running the simulator.
* Design, train and validate a convolutional neural network model that predicts a steering angle from image data.
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.

### Files submitted

* model.py : This file contains the neural network implementation.
* drive.py : This file contains the code to drive the car autonomously in the simulator based on the model. This file is provide by Udacity. I haven't made any changes to this file.
* model.h5 : model file that contains the learned weights of the neural network.
* video.mp4 : Video of the car running in first track of the simulator in autonomous mode.
* prepare_data.py : Unzip the Udacity provided data.zip file and copy the image files to /opt/ directory for training.

### Quality of Code

I am able to drive the car autonomously in the Udacity simulator using the trained model successfully. `video.mp4` contains a video of the car running in the simultaor autonomously

Code is organized in to functions and code is reasonably commented to make it readable.


### Model Architecture and Training Strategy

#### Architecture

My implementation of the CNN is based on NVIDIA convolutional neural network as suggested in the Udacity project instructions. Details of the architecture can be found [here.](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) NVIDIA network has about 27 million connections and 250 thousand parameters. See the architecture of NVIDIA CNN below.

![alt text][image1]

Input size of the above neural net is 66x200x3. The original images produced by our car camera is 160x320x3. During the initial stages of neural net, the upper and some lower part of the image was removed. This is accomplished by the following lines of code.

```python
# Crop the image to remove irrelevant part of the image
model.add(Cropping2D(cropping=((70,25),(0,0))))  
```

This makes the size of the images to 65x320x3. I have decided use this cropped image as input to first convolutional layer.

I tried to generate my own training data by running the car in first track of the Udacity simulator. I found that controlling the car in the track and generating the data quite challenging. I found that the data provided by Udacity [here](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) is reasonably good. So decided to try to train the network using the Udacity provided data and then generate my own data only if it is required later. 

#### Data Generator
A `generator()` function was implemented in `model.py` to return the data in batches to the model during training to reduce the memory usage. This generator function shuffles the data so that the order in which images appear in the samples doesn't affect the neural network. Car is mounted with three cameras: center, left and right. Each record in the csv file contains these three camera snapshots and the corresponding steering angle. Since there is only one steering angle associated with three images, I have decided to apply a correction factor of 0.2. For left images, steering angle is increased by 0.2 and for right images, steering angle is decreased by 0.2.

#### Data Augmentation
The data provided by Udacity is captured by running the car in Simulator. Additional data is generated by flipping the data horizontally as suggested in the lessons. This is implemented in `generator()` in `model.py`. Image flipping is implemented using the OpenCV function `cv2.flip()`. Steering angle is multiplied by -1 for the flipped images.

Another technique suggested in the lesson to avoid bias was to drive car in opposite direction in test track, but I haven't done it.


#### Training and Validation data

I have used the data provided by Udacity for training and validation of the neural network. I have used sklearn's `train_test_split()` function to split the give data to training (80%) and validation(20%) set. 

Following is an example training image from Udacity provided data set.

![alt text][image3]


#### Final Architecture 

Here is the summary of the neural network generated by `model.summary()`.

![alt text][image2]

Network layers

1. First lambda is used to normalize the image.
2. Cropping was applied to remove the sky/trees from top of the image and dash board from the bottom of the image.
3. First convolution layer with filter depth of 24, filter depth (5x5) and stride=(2x2)
4. 'Relu' activation function
5. Second convolution layer with filter depth of 36, filter size of (5x5) and stride of (2x2)
6. 'Relu' activation function
7. Drop out layer with 25% dropout
8. Third convolution layer with filter depth of 48, filter size of (5x5) and stride of (2x2)
9. 'Relu' activation function
10. Drop out layer with 25% dropout
11. Fouth convolution layer with filter depth of 64, filter size of (3x3) and stride of (1x1)
12. 'Relu' activation function
13. Fifth convolution layer with filter depth of 64, filter size of (3x3) and stride of (1x1)
14. 'Relu' activation function
15. Flatten to feed to fully connected layers. Output size=2112
16. First fully connected layer. Output size=100
17. 'Relu' activation function
18. Drop out layer with 25% dropout
19. Second fully connected layer. Output size=50
20. 'Relu' activation function
21. Drop out layer with 25% dropout
22. Third fully connected layer. Output size=10
23. 'Relu' activation function
24. Fourth fully connected layer. Output size=1

Final output computes the steeing angle. Since this is a regression (and NOT classification) we are using "mean square error" as the loss function and adam optimizer.

```python
model.compile(loss='mse',optimizer='adam')
```

Dropout layers have been added to reduce overfitting of the data.



