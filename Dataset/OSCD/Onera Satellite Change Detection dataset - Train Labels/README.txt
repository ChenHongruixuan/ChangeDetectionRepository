Onera Satellite Change Detection dataset

##################################################
Authors: 
Rodrigo Caye Daudt, rodrigo.daudt@onera.fr
Bertrand Le Saux, bertrand.le_saux@onera.fr
Alexandre Boulch, alexandre.boulch@onera.fr
Yann Gousseau, yann.gousseau@telecom-paristech.fr

##################################################
About:
This dataset contains registered pairs of 13-band multispectral satellite images obtained by the Sentinel-2 satellites of the Copernicus program. Pixel-level urban change groundtruth is provided. In case of discrepancies in image size, the older images with resolution of 10m per pixel is used. Images vary in spatial resolution between 10m, 20m and 60m. For more information, please refer to Sentinel-2 documentation.

The proposed split into train and test images is contained in the train.txt and test.txt files.

For downloading and cropping the images, the Medusa toolbox was used.
https://github.com/aboulch/medusa_tb

For precise registration of the images, the GeFolki toolbox was used.
https://github.com/aplyer/gefolki

##################################################
Labels:
The train labels are available in two formats, a .png visualization image and a .tif label image. In the png image, 0 means no change and 255 means change. In the tif image, 0 means no change and 1 means change. 

<ROOT_DIR>/<CITY>/cm/ contains:
- cm.png
- <CITY>-cm.tif

Please note that prediction images should be formated as the <CITY>-cm.tif rasters for upload and evaluation on DASE (http://dase.ticinumaerospace.com/).

##################################################
Citation:
If you use this dataset for your work, please use the following citation:

@inproceedings{daudt-igarss18,
author = {{Caye Daudt}, R. and {Le Saux}, B. and Boulch, A. and Gousseau, Y.},
title = {Urban Change Detection for Multispectral Earth Observation Using Convolutional Neural Networks},
booktitle = {IEEE International Geoscience and Remote Sensing Symposium (IGARSS'2018)},
venue =  {Valencia, Spain},
month = {July},
year = {2018},
}

##################################################
Copyright:
Sentinel Images:
This dataset contains modified Copernicus data from 2015-2018. Original Copernicus Sentinel Data available from the European Space Agency (https://sentinel.esa.int).

Change labels:
Change maps are released under Creative-Commons BY-NC-SA. For commercial purposes, please contact the authors.

