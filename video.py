import torch
import bbox_visualizer as bbv
import cv2
import sys
import os
import argparse
import numpy as np

labels = ['Right_Scissors',
          'Left_Scissors',
          'Right_Needle_driver',
          'Left_Needle_driver',
          'Right_Forceps',
          'Left_Forceps',
          'Right_Empty',
          'Left_Empty',
          '0']

idx_to_label = {i: inst for i, inst in zip(range(len(labels)), labels)}
label_to_idx = {inst: i for i, inst in zip(range(len(labels)), labels)}
label_to_idx['0'] = -1
class_appearance = {i: 0 for i in range(len(labels))}
tool_usage_right = {"Right_Empty": "T0",
                    "Right_Needle_driver": "T1",
                    "Right_Forceps": "T2",
                    "Right_Scissors": "T3"}

tool_usage_left = {"Left_Empty": "T0",
                   "Left_Needle_driver": "T1",
                   "Left_Forceps": "T2",
                   "Left_Scissors": "T3"}

tool_usage_right = {v: k for k, v in tool_usage_right.items()}
tool_usage_left = {v: k for k, v in tool_usage_left.items()}


def draw_text(img, text,
              font=cv2.FONT_HERSHEY_PLAIN,
              pos=(0, 0),
              font_scale=3,
              font_thickness=2,
              text_color=(0, 255, 0),
              text_color_bg=(0, 0, 0)
              ):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)
    return text_size


def parse_args():
    parser = argparse.ArgumentParser()
    required = parser.add_argument_group('required arguments')

    required.add_argument("--video_path", required=True,
                          help="Full paths to images to predict tool usage on",
                          dest="image_path")
    required.add_argument("--yolo_path", required=False,
                          help="Full paths to the model.pt file containing the YOLO weights",
                          dest="yolo_path",
                          default='yolov5/runs/train/exp11/weights/best.pt')

    required.add_argument("--gt", required=False,
                          help="Full paths to the model.pt file containing the YOLO weights",
                          dest="gt",
                          default=None)
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


def label_video(video_path, model, gt=None):
    vid_name, vid_ext = video_path.split(".")[0], video_path.split(".")[1]
    base_name = os.path.basename(vid_name)
    print(f"Labeling video: {vid_name}")
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    if gt:
        try:
            with open(gt + f"/tools_right/{base_name}.txt", "r") as f:
                vid_right_labels = [line.split(" ") for line in f.readlines()]

            with open(gt + f"/tools_left/{base_name}.txt", "r") as f:
                vid_left_labels = [line.split(" ") for line in f.readlines()]
        except Exception:
            print("Error with ground truth labels, not implementing gt labeling")
    cap = cv2.VideoCapture(video_path)
    label_path = vid_name + "_labeled." + vid_ext
    out = cv2.VideoWriter(label_path, fourcc, 30.0,
                          (int(cap.get(3)), int(cap.get(4))))

    if not cap.isOpened():
        print("Error opening video stream or file")

    left_running_average = []
    right_running_average = []

    window_span = 10
    final_predictions = {"Left_pred_after": [], "Right_pred_after": [], "Left_gt": [], "Right_gt": [],
                         "Right_pred_before": [],
                         "Left_pred_before": []}

    i = 0
    while cap.isOpened():
        i += 1
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_res = model(frame)
            bbox = frame_res.pandas().xyxy[0][['xmin', 'ymin', 'xmax', "ymax"]].astype(int).values.tolist()
            classes = frame_res.pandas().xyxy[0][['confidence', 'name']].values.tolist()
            results = frame_res.pandas().xyxy[0]
            before_labels = [bla[1] for bla in enumerate(classes)]
            try:
                final_predictions['Right_pred_before'].append(
                    [rlabel[1] for rlabel in before_labels if "Right" in rlabel[1]][0])
            except Exception:
                final_predictions['Right_pred_before'].append('0')
            try:
                final_predictions['Left_pred_before'].append(
                    [rlabel[1] for rlabel in before_labels if "Left" in rlabel[1]][0])
            except Exception:
                final_predictions['Left_pred_before'].append('0')

            if len(results) == 0:
                right_running_average.append('0')
                left_running_average.append('0')
                right_box = -1
                left_box = 0
            elif len(results) == 1:
                pred = results['name'].iloc[0]
                if 'Right' in pred:
                    right_running_average.append(pred)
                    left_running_average.append('0')
                    right_box = 0
                    left_box = -1
                else:
                    right_running_average.append('0')
                    left_running_average.append(pred)
                    right_box = -1
                    left_box = 0
            else:
                right_running_average.append(results[results['xmin'] == min(results['xmin'])]['name'].iloc[0])
                right_box = list(results[results['xmin'] == min(results['xmin'])].index)[0]
                left_running_average.append(results[results['xmin'] == max(results['xmin'])]['name'].iloc[0])
                left_box = list(results[results['xmin'] == max(results['xmin'])].index)[0]

            if i < window_span:
                left_label = left_running_average[-1] if left_running_average[-1] != '0' else 'Left_Empty'
                right_label = right_running_average[-1] if right_running_average[-1] != '0' else 'Right_Empty'
                final_predictions['Left_pred_after'].append(left_label)
                final_predictions['Right_pred_after'].append(right_label)

            if len(right_running_average) == window_span:
                left_label = max(set(left_running_average), key=left_running_average.count)
                right_label = max(set(right_running_average), key=right_running_average.count)
                final_predictions['Left_pred_after'].append(left_label)
                final_predictions['Right_pred_after'].append(right_label)
                right_running_average = right_running_average[1:]
                left_running_average = left_running_average[1:]
            if gt:
                try:
                    for t in vid_right_labels:
                        if int(t[0]) <= i <= int(t[1]):
                            final_predictions['Right_gt'].append(tool_usage_right[t[2].strip()])

                    for t in vid_left_labels:
                        if int(t[0]) <= i <= int(t[1]):
                            final_predictions['Left_gt'].append(tool_usage_left[t[2].strip()])
                except Exception as e:
                    print(e)
                    break

            classes = [f"{label}, {round(conf,3)}" for conf, label in classes]
            frame = bbv.draw_multiple_rectangles(frame, bbox, bbox_color=(0, 255, 0))
            frame = bbv.bbox_visualizer.add_multiple_T_labels(frame, classes, bbox, draw_bg=True,
                                                              text_bg_color=(255, 255, 255), text_color=(0, 0, 0))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            color = (255, 0, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1
            fontScale = 0.5
            org = (10, 15)
            try:
                gt_text = f"Ground truth: {final_predictions['Right_gt'][-1]} and {final_predictions['Left_gt'][-1]}"
                draw_text(frame, gt_text, pos=(10, 15), font_scale=1, font_thickness=thickness)
                pred_text = f"Prediction: {final_predictions['Right_pred_after'][-1]} and {final_predictions['Left_pred_after'][-1]}"
                draw_text(frame, pred_text, pos=(10, 30), font_scale=1, font_thickness=thickness)
            except Exception:
                pred_text = f"Prediction: {final_predictions['Right_pred_after'][-1]} and {final_predictions['Left_pred_after'][-1]}"
                draw_text(frame, pred_text, pos=(10, 15), font_scale=1, font_thickness=thickness)
            bla = out.write(frame)
        else:
            break

    cap.release()
    out.release()


def main():
    args = parse_args()
    path_to_vid_to_predict = args.image_path
    if not os.path.isfile(path_to_vid_to_predict):
        print(f"Path given: {path_to_vid_to_predict}")
        raise FileExistsError("Path to picture is invalid")
    print(f"Predicting tool usage on Video: {path_to_vid_to_predict}")
    yolo = load_model(args.yolo_path)
    gt = args.gt
    label_video(path_to_vid_to_predict, yolo, gt)


if __name__ == '__main__':
    main()
