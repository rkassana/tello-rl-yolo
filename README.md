# tello-rl-yolo
Tello drone implementation with YOLO and DDPG control.


![](demo2.gif)


This capstone project was realized in the context of the Udacity Machine Learning Nanodegree.

Dataset:

VOC2012 Train/val : http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar

VOC2012 Test : http://pjreddie.com/media/files/VOC2012test.tar

Inspired from :

- Keras YOLO V3 implementation : https://github.com/experiencor/keras-yolo3

- Tello Python wrapper : https://github.com/damiafuentes/DJITelloPy

- Drone tracking (DDPG) : Keras-rl / rkassana

# How To #

Requirements :
- Python 3.X
- Keras GPU
- Keras-rl
- OpenCV
- Numpy
- CUDA & NVIDA Drivers
- OpenAI Gym

Make sure YOLO weight file VOC.h5 is in the root folder : https://drive.google.com/open?id=15oONh_eIdz3CkHdwybZeB49rDCpE0X9A


1- Start main.py

2- Once the video is on, it will take 30 seconds for the YOLO and DDPG to initialize (model creation, loading, etc..).

3- Take off using T

4- Drone should track bounding box in screen.

