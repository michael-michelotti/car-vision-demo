# car-vision-demo
Demonstration of script to use Computer Vision (CV) to identify cars from a video stream
![Car video with detection](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExcWluODNnaW85OXQzcjludWk3NWMxNXk3czR5dzNhZHo0OTR0YjlkOCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/g530zImi4UNcO00rvq/giphy.gif)

## Overview
This is a demonstration script utilizing the [OpenCV](https://opencv.org/) Python API to draw Computer Vision (CV) prediction boxes around cars. The input car video is provided as cars_at_intersection.MOV. 

## Getting Started
### Prerequisites
#### Hardware
No hardware is required.
#### Software
- Python Interpreter ([CPython 3.10](https://www.python.org/downloads/release/python-3100/) was used in this case)
- YOLO3 configuration and weights
### Installation
__1. Download and install Python interpreter__

__2. Download the source code__
```
git clone git@github.com:michael-michelotti/car-vision-demo.git && cd car-vision-demo
```

__3. Download YOLO3 configuration and weights__

Download the Darknet-trained YOLO3 model (trained on COCO dataset) and configuration from [the YOLO website](https://pjreddie.com/darknet/yolo/). Scroll down to YOLOv3-320 and download from the `cfg` and `weights` links.

__4. Install dependencies (you may want to create a virtual environment first; `python -m venv .venv`)__
```
pip install -r requirements.txt
```

__5. Run the script!__
```
python main.py
```

## Usage
Once you've installed the dependencies and run the script, you should have a video window open playing the car video with predictions included.

You can copy any new video into the project directory and adjust the `VIDEO_FILENAME` parameter to process a new video.

You can adjust the desired output FPS by changing the OUTPUT_FPS parameter.

You can adjust the box color between CV2_BLUE, CV2_RED, and CV2_GREEN using the BOX_COLOR parameter.

Currently, the script is configured to use your CPU for DNN model predictions. It can be adjusted to use a GPU as a backend, but it would require installation of [CUDA](https://developer.nvidia.com/cuda-downloads), [cuDNN](https://developer.nvidia.com/cudnn-downloads), and a [specially compiled version of the OpenCV library](https://github.com/cudawarped/opencv-python-cuda-wheels/releases).
