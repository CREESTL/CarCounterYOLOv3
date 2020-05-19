# CarCounterYOLOv3
### How to run
- Download `.weights` file for YOLO [here](https://yadi.sk/d/rrlsHZFHyPmnCA)
- Download any test-video with cars driving around and put it to `videos/` folder
- Move `.weights` file to `yolo/` folder
- Go to the project's repository via command line
- type `python car_counter_yolov3.py -y yolo --input videos/THE_NAME_OF_YOUR_TEST_VIDEO --output output --skip-frames 5` and hit `Enter`

You can change the `skip-frames` parameter (the higher it is, the faster the program works). But the accuracy will be lower

The proccessed video will be saved to the `output/` folder
