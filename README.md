# Segmentation Zoo

## This library contains the code to perform segmentation with the following networks:

FCN-8
FCN-16
FCN-32
U-Net
SegNet

The data should be stored as follows: <br />

Within `Segmentation_parameters.py`, specify the `data_root`. Within this directory store Images, labels and weights as the following directories:

`Images` <br />
`Labels` <br />
`Weights` <br />

Labels are stored as rgb images with 3 channels where the first channel gives the positive class and the second channel gives the negative class. Weights have one channel. Corresponding image, label and weight files must have the same name. There must be the same amount of data within each folder and all data should have the same x,y dimensions. 

To start training type within the terminal: <br />

`python Segmentation_main.py train`






