# web streaming

This repository contains a demonstration of video streaming over API using flask framework and pertained  YOLO V3

## How to use

To install all the requirements for the project run

	pip install -r requirements.txt

Now run the weights_converter.py file **(IMPORTANT: ONLY RUN THIS FILE ONCE)**.
    
    python weights_converter.py
     
In the root directory. After the modules have been installed you can run the project by using python

	python webstreamer.py
	
Open http://localhost:8000 in a browser.you can open multiple tabs as this application support multi 
threading architecture.

**Note : Do not use dockerfile cause work is still in progress**