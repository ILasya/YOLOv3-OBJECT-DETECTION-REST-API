# OBJECT-DETECTION-using-YOLOv3
An Object Detection Application using YOLOv3 (PyTorch and Django Implementation).
Django implementation for Webpage as well as REST API

# Introduction

A web application that provides object detection using YOLOv3 and also generates REST API. It's implemented using Django framework and PyTorch (for YOLO model).
I have developed Django API which accepts an image as request. The input image is converted to float32 type NumPy array and passed on to the YOLOv3 object detection model. The model performs object detection for the image and generates a JSON object with names of all the objects and their respective frequency in the image. The Response of the API is the JSON object.

# Required Libraries

The libraries required along with thier version are mentioned below:
* Python  (3.7)
* Django  (3.0.3)
* PyTorch (1.3.1)
* Pillow  (7.1.2)
* OpenCV  (4.2.0)
* NumPy   (1.18.5)

# Required files for Detection

For object detection using Pre-Trained model, we need three important files :
* yolov3.cfg - The cfg file describes the layout of the network, block by block. The official cfg file is available in darknet github repository. However, I have made few changes in the configuration file in order to get better performance.
* yolov3.weights - We use weights from the darknet53 model.
* coco.names - The coco.names file contains the names of the different objects that our model has been trained to identify.

# Steps to Follow (Working)

This repository can do two things:
1. Implementation web-application
2. Generation of REST API (API testing is done using POSTMAN)

##	1) Webpage Implementation

1.	Clone the GitHub repository

2. Change the directory to the cloned Repository folder.

3.	The .cfg and coco.names files are already available in this repository. Now, all we need to do is download the weights file.

Download yolov3.weights using the following command in your Command Prompt:

```wget https://pjreddie.com/media/files/yolov3.weights```
 
4.	Install all the required libraries.

5.	Execute the code below: (This command needs to be executed only once) 

```python manage.py collectstatic```

This command initiates Django and collects all the static files.

6.	Then, Execute: 

```python manage.py runserver```

This command starts Django server. Now we are all set to run the application.

7.	After executing the above code, you will see something like this:

![image](https://user-images.githubusercontent.com/54140890/90550933-436d0a00-e1ae-11ea-9f08-bb2858a2c75b.png)

8.	Click on the link. This will direct you to the web browser. 

9.	Select an image via Drag-&-Drop or Browse mode.

![image](https://user-images.githubusercontent.com/54140890/90551081-731c1200-e1ae-11ea-98f2-5fe3f13eb1a4.png)

10.	Click on ”Detect Object”

![image](https://user-images.githubusercontent.com/54140890/90551134-84fdb500-e1ae-11ea-9e81-ada002e65738.png)

![image](https://user-images.githubusercontent.com/54140890/90551194-9c3ca280-e1ae-11ea-85b7-24f4be819454.png)

11.	Input to Django Web-app is an image. This input image is converted to float32 type NumPy array and passed on to the YOLOv3 model. The model performs object detection for the image and generates a JSON object with names of all the objects and their respective frequency in the image. 

![image](https://user-images.githubusercontent.com/54140890/90551297-bc6c6180-e1ae-11ea-8434-878ef9930da7.png)

12.	The Form Response is the JSON object. This JSON object is displayed as shown above.

13.	Now, click on “Show Predictions” to see the image with Bounding Boxes.

![image](https://user-images.githubusercontent.com/54140890/90551388-dc9c2080-e1ae-11ea-988e-cc2465cc0d49.png)
 
14.	 To try on other images, click on “Choose a New File”.


## 2) REST API Implementation - POSTMAN

Postman is a scalable API testing tool. The steps to be followed are:

1.	Follow the first 6 steps as mentioned above.

2.	Make sure the server runs properly

![image](https://user-images.githubusercontent.com/54140890/90551526-18cf8100-e1af-11ea-802c-7ee553849014.png)

3.	Open POSTMAN and select POST option. Enter the server link shown above and append /object_detection/api_request/ to it. 

For Example : 127.0.0.1:8000/object_detection/api_request/

![image](https://user-images.githubusercontent.com/54140890/90551940-b5921e80-e1af-11ea-9d8d-50d666c7983e.png)

4.	Click on Body. Enter key value as “Image”. Choose the image file and click on “SEND”.

5.	The input image is converted to float32 type NumPy array and passed on to the YOLOv3 model. The model performs object detection for the image and generates a JSON object with names of all the objects and their respective frequency in the image. 

![image](https://user-images.githubusercontent.com/54140890/90552102-f4c06f80-e1af-11ea-9890-0d24d11a2ea5.png)

6.	The HttpResponse is the JSON object. This JSON object is displayed as shown above.
