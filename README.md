# YOLOv5-Tool-Detection-HW1
## Goal
Use YOLOv5 to predict tool usages of physicians performing practice surgury operations.<br>

## Description
<ul>
  <li> Used [RoboFlow](https://app.roboflow.com) for data analysis and health checks.  </li>
  <li> Used the PyTorch implementation of [YOLOv5](https://github.com/ultralytics/yolov5) to predict the tool usages of physicians when performing surgical activity. 
 </li>
  <li> Implementation of smoothing mechanism for video object classification.</li>
  <li> Reporting evaluation metrics.</li>
 </ul>

## How to run?
#### Prepare envoriment
1. Clone this project
2. pip/conda install the requirments.txt file

### Reproduce results
#### main.py
The file which does all of the heavy lifting is `main.py`. <br>
`main.py` is responsible for annotating videos with bounding boxes, classes and confidences. It also reports on the video the current tool usage using a argmax smoothing mechanism. Finally it reports $f_{1}$ score, $Precision$ and $Recall$. <br>
`main.py` assumes the following dir structure:
```
root
|-> main.py
|-> videos/
|--->video_1.wmv
|--->video_2.wmv
|      .
|      .
|      .
|--->tools_right/
|------>video_1.txt
|------>video_2.txt
|      .
|      .
|      .
|--->tools_left/
|------>video_1.txt
|------>video_2.txt
```
where `videos` is a directory in the same dir as `main.py` and it contains video files. In the `videos` directory there are two other directories called `tools_left` and `tools_right`, each contaning a corrosponding text file with the same name as a video in `videos` dir and the text files contain the labels in frame format:
```
0 308 T0
309 674 T2
675 1649 T0
1650 2126 T2
2127 3084 T0
3085 3338 T2
3339 4450 T0
```
it can be run using this command:
```
python main.py
```
#### predict.py
The purpose of `predict.py` is to detect tools in a single frame. Given a frame it will save in the same directory the same frame with bounding boxes, classifications and confidence in each class.<br>
It is run using the following command:
```
python predict.py --image_path path/to/image.jpg
```
Other optional parameters are `yolo_path` which specifies which pretrained model to use to make predictions. The default value is `yolov5/runs/train/exp11/best.pt` For example:
```
python predict.py --image_path path/to/image.jpg --yolo_path yolov5/runs/train/exp12/best.pt
```
After running this command a new labeled image will be created in the same dir as `image_path` 
#### video.py
The purpose of `video.py` is to perform object detection and tool usage on videos of educational surgical activity.<br>
`video.py` can be run using the following command:
```
python video.py --video_path path/to/videos/video.wmv
```
Other optional parameters include:<br>
`--yolo_path` - Path specifying pretrained weights to use to perform predictions.<br>
`--gt` Path to ground truth labels in the same format as the file `main.py` expects (same dir structre).<br>
For example:
```
python video.py --video_path path/to/videos/video.wmv --yolo_path path/to/yolo_weights.pt --gt path/to/videos/
```
This module will save in the same directory as `video.wmv` a new labeled video file. 
