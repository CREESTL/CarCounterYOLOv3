# Quick Overview
In this repository you can see 2 main programs: `car_counter_yolov3_custom_classes.py` and `car_counter_yolov3_COCO_6_classes.py`

The first one is a lighter version of the second. Basically, I`ve trained YOLOv3 to detect 5 classes:
- sedan
- minivan
- SUV
- hatchback
- universal
But, to be honest, `.weights` file that I got in the end is pretty wack and works not that good on different videos. But it's still here.
### How to run it
- Download `.weights` file for YOLO [here](https://yadi.sk/d/rrlsHZFHyPmnCA)
- Download any test-video with cars driving around and put it to `videos/` folder (or use any of those that are already there)
- Move `.weights` file to `yolo/` folder
- Go to the project's repository via command line
- type `python car_counter_yolov3_custom_classes.py -y yolo --input videos/THE_NAME_OF_YOUR_TEST_VIDEO --output output --skip-frames 5` and hit `Enter`


The proccessed video will be saved to the `output/` folder

The second one uses pretrained `.weights` file from [this site](https://pjreddie.com/darknet/yolo/). So I didn't need to train YOLOv3 myself once again. This program can:
- detect and track objects of all of 80 COCO classes
- count objects of each of 6 classes:
    - car
    - truck
    - person
    - motorcycle
    - bicycle
    - bus
- count the amount of all of those objects on each frame of the video
- put the results into `.json` file

  
  ### How to run it
- Download `YOLOv3-608.weights` file for YOLO [here](https://pjreddie.com/darknet/yolo/)
- Download any test-video with cars driving around and put it to `videos/` folder (or use any of those that are already there)
- Move `.weights` file to `yolo/` folder
- Go to the project's repository via command line
- type `python car_counter_yolov3_COCO_6_classes.py -y yolo --input videos/THE_NAME_OF_YOUR_TEST_VIDEO --output output --skip-frames 5` and hit `Enter`

  You can change the `skip-frames` parameter (the higher it is, the faster the program works). But the accuracy will be lower

  The proccessed video and the `.json` file will be saved to the `output/` folder
