import torch
import bbox_visualizer as bbv
import cv2
import sys
import os
import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')

    required.add_argument("--image_path", required=True,
                          help="Full paths to images to predict tool usage on",
                          dest="image_path")
    required.add_argument("--yolo_path", required=False,
                          help="Full paths to the model.pt file containing the YOLO weights",
                          dest="yolo_path",
                          default='yolov5/runs/train/exp11/weights/best.pt')
    options = parser.parse_args()
    return options


def load_model(yolo_path):
    answer = "Yes" if torch.cuda.is_available() else "No"
    print(f"CUDA available?: {answer}")
    model = torch.hub.load('yolov5', 'custom', path=yolo_path, source='local')
    if torch.cuda.is_available():
        model.cuda()
    model.max_det = 2  # maximum number of detections per image
    model.amp = True  # Automatic Mixed Precision (AMP) inference
    return model


def main():
    args = parse_args()
    path_to_pic_to_predict = args.image_path
    if not os.path.isfile(path_to_pic_to_predict):
        print(f"Path given: {path_to_pic_to_predict}")
        raise FileExistsError("Path to picture is invalid")
    print(f"Predicting tool usage on image: {path_to_pic_to_predict}")
    yolo = load_model(args.yolo_path)
    frame = cv2.imread(path_to_pic_to_predict)
    results = yolo(frame)
    print("Bounding boxes:")
    print(results.pandas().xyxy[0])
    file_name, file_ext = path_to_pic_to_predict.split(".")[0], path_to_pic_to_predict.split(".")[-1]
    new_file_path = file_name + "_labeled." + file_ext
    new_file_path = r"{}".format(new_file_path)
    print(f"Trying to save labeled image to {new_file_path}")
    frame = np.squeeze(np.asarray(results.render()), 0)
    if not cv2.imwrite(new_file_path, frame):
        raise Exception("Could not save image")
    print(f"Labeled image saved in {new_file_path}")


if __name__ == '__main__':
    main()
