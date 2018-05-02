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
`Weights`

To start training type within the terminal: <br />

`Segmentation_main.py train`






