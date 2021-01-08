# web streaming

This repository contains a demonstration of video streaming over API using flask framework

## How to use

To install all the requirements for the project run

	pip install -r requirements.txt

Now download the model required to run the VGG16 net **(IMPORTANT)**.
    
     https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
     
after downloading place the model in /model/weights dir in present in project structure (i.e if /weights dir is not present then create the dir)

In the root directory. After the modules have been installed you can run the project by using python

	python webstreamer.py
	
Open http://localhost:8000 in a browser.you can open multiple tabs as this application support multi 
threading architecture.

**Note : Do not use dockefile cause work is still in progress**