# Behavioral Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
This repository contains the file for the Behavioral Cloning Project.

In this project, i learned about deep neural networks and convolutional neural networks to clone driving behavior. I train, validate and test the neuronal network model using Keras. The model outputs a steering angle to an autonomous vehicle.

In a provieded simulator data could be collected. The image data and steering angles from the data is used train a neural network and then use this model to drive the car autonomously around the track. Amazing.

Following five files are stored in the repository:

* model.py (script used to create and train the model)
* drive.py (script to drive the car - feel free to modify this file)
* model.h5 (a trained Keras model)
* a report writeup file (either markdown or pdf)
* video.mp4 (a video recording of your vehicle driving autonomously around the track for at least one full lap)

This README file describes how to output the video in the "Details About Files In This Directory" section.

The Project
---
The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report


## 1. Data collection

To collect the data, the vehicle is driving for one cource in the simulation. To extend the collect able data, the cource is recorded with three comeras. The mounting posistions are seen in Fig. 1.1.  To train the model later, every image sequence is saved the current steering angle. So that the model can learn the posistion relativ to the steering angle. To compensate the offset between the side cameras and the steering angle, seen on the top right side if Fig. 1.1,  an addiational offset ```+_20%```. 


<figure>
 <img src="./examples/Kamera_Offset_png'.png" width="650" alt="data amout plot" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 1.1: Visualization camera positions</p> 
 </figcaption>
</figure>
 <p></p>

<figure>
 <img src="./examples/LeftCenterRightImage.jpg" width="650" alt="data amout plot" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 1.2: Data Collegetion</p> 
 </figcaption>
</figure>
 <p></p>

The data argumentations process and preprocessing can be seen in Fig. 1.3. The data gets extended by flipping the images and blurring the images. so that in the end 1/3 of the data is from the trac collected, 1/3 fliped and the rest blurred. This is done due to generalize the model more. To excelerate the training process, the unsecessary parts of the image that contain any relevant inforamtion get  croppd. So that the model have to process less image data. Seen on the very right of Fig 1.3.

<figure>
 <img src="./examples/PreProcess.jpg" width="650" alt="data amout plot" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 1.3: Data Collegetion</p> 
 </figcaption>
</figure>
 <p></p>

## 2. Model

My final model results were:
* training set accuracy of 99.4%
* validation set accuracy of 93.8%
* test set accuracy of 91.9%

**Architecture** As archtitecture has been choosen the architecture from the Nvidia End to End Learning [Paper](https://arxiv.org/abs/1604.07316). The use case is the same and there for its convinient to beginn with the same aritecture and than see if the net have to be modifiyed. But since its the same use case and the same data no big modifications have to be done. As modification a cropped layer and three dropout layers are added at the fully conntected layers. Fig. 

<figure>
 <img src="./examples/Architecture.jpg" width="850" alt="data amout plot" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 2.1: Net</p> 
 </figcaption>
</figure>
 <p></p>

**Model Parameters** 
The model is trained with following parameters:

* EPOCHS = 7
* BATCH_SIZE = 512
* dropout = 0.5
* learning rate = 0.001
* optimizer: AdamOptimizer
* 20% data splite for validation

The Training too place on an external GPU. Therefore the Batch Size have been optimaized to the highes possible number, before the Computer crashes, due the face its running out of memory. The Dropuout rate is the lowest recomended but it leeds still to an ligth overfitting since validation and training accourancy divergent, seen in Fig. 2.2. This also shows us that the learning rate could be reducecould, so that the net is more generelazing.

<figure>
 <img src="./examples/PreProcess.jpg" width="850" alt="data amout plot" />
 <figcaption>
 <p></p> 
 <p style="text-align: center;"> Fig. 2.2: Training and Validation accourancy.</p> 
 </figcaption>
</figure>
 <p></p>


**Problem Solving Approach**  An iterativ approach have been taken. The model archtitecture obove have been choosen and it started by training the model with the data of one track and one round. After every of the following steps the training and validaion accurany have been alaysed as well as the autonomous drining perfomrance on the training. 

1. Then the learning rate was reduced due to high overfitting. 
2. The training data was argumented, to reduce the overfitting. This increaced the validation accurancy a lot. But still the vehicle could not performe on autonomous track. 
3. Including three dropout layer at the fully connected layer. This increaced the validation accurancy much more and let the vehicle drive the track autonomouly.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab enviroment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.

## Details About Files In This Directory

### `drive.py`

Usage of `drive.py` requires the saved trained model as an h5 file, i.e. `model.h5`.  Once the model has been saved, it can be used with drive.py using this command:

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

### Hint
- Please keep in mind that training images are loaded in BGR colorspace using cv2 while drive.py load images in RGB to predict the steering angles.

