import pandas as pd
import torch
import bbox_visualizer as bbv
import cv2
import os
from sklearn.metrics import precision_recall_fscore_support, f1_score, accuracy_score
from os import listdir
from os.path import isfile, join


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


answer = "Yes" if torch.cuda.is_available() else "No"
print(f"CUDA available?: {answer}")
model = torch.hub.load('yolov5', 'custom', path='yolov5/runs/train/exp11/weights/best.pt', source='local')
if torch.cuda.is_available():
    model.cuda()
model.max_det = 2  # maximum number of detections per image
model.amp = True  # Automatic Mixed Precision (AMP) inference

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

videos = [f for f in listdir("videos") if isfile(join("videos", f)) if "wmv" in f]
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

before_pred = []
after_pred = []
gt = []

for vid in videos:
    vid_name = vid.split(".")[0]
    print(f"Labeling vidoe: {vid_name}")
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter("videos/" + vid_name + "_labeled.wmv", fourcc, 30.0, (640, 480))
    try:
        with open(f"videos/tools_right/{vid.split('.')[0]}.txt", "r") as f:
            vid_right_labels = [line.split(" ") for line in f.readlines()]

        with open(f"videos/tools_left/{vid.split('.')[0]}.txt", "r") as f:
            vid_left_labels = [line.split(" ") for line in f.readlines()]
    except Exception:
        print("Skipping")
        continue
    cap = cv2.VideoCapture("videos/" + vid)

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

            for t in vid_right_labels:
                if int(t[0]) <= i <= int(t[1]):
                    final_predictions['Right_gt'].append(tool_usage_right[t[2].strip()])

            for t in vid_left_labels:
                if int(t[0]) <= i <= int(t[1]):
                    final_predictions['Left_gt'].append(tool_usage_left[t[2].strip()])

            classes = [f"{label}, {conf}" for conf, label in classes]
            frame = bbv.draw_multiple_rectangles(frame, bbox, bbox_color=(0, 255, 0))
            frame = bbv.bbox_visualizer.add_multiple_T_labels(frame, classes, bbox, draw_bg=True,
                                                              text_bg_color=(255, 255, 255), text_color=(0, 0, 0))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            color = (255, 0, 0)
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = 1
            fontScale = 0.5
            org = (10, 15)
            gt_text = f"Ground truth: {final_predictions['Right_gt'][-1]} and {final_predictions['Left_gt'][-1]}"
            pred_text = f"Prediction: {final_predictions['Right_pred_after'][-1]} and {final_predictions['Left_pred_after'][-1]}"

            draw_text(frame, gt_text, pos=(10, 15), font_scale=1, font_thickness=thickness)
            draw_text(frame, pred_text, pos=(10, 30), font_scale=1, font_thickness=thickness)
            out.write(frame)
        else:
            break

    cap.release()
    out.release()

    print(vid_name + " Predictions:")
    print("\tRight predictions")
    os.mkdir("videos/" + vid_name)
    vid_name = "videos/" + vid_name + "/results"
    os.mkdir(vid_name)
    vid_name += "/"

    f1_macro_before = f1_score(final_predictions['Right_gt'] + final_predictions['Left_gt'],
                               final_predictions['Right_pred_before'] + final_predictions['Left_pred_before'],
                               labels=labels,
                               average='macro')

    f1_macro_after = f1_score(final_predictions['Right_gt'] + final_predictions['Left_gt'],
                              final_predictions['Right_pred_after'] + final_predictions['Left_pred_after'],
                              labels=labels,
                              average='macro')

    accuracy_before = accuracy_score(final_predictions['Right_gt'] + final_predictions['Left_gt'],
                                     final_predictions['Right_pred_before'] + final_predictions['Left_pred_before'])

    accuracy_after = accuracy_score(final_predictions['Right_gt'] + final_predictions['Left_gt'],
                                    final_predictions['Right_pred_after'] + final_predictions['Left_pred_after'])

    with open(vid_name + "f1_macro.txt", 'w') as f:
        f.write(f"f1_macro_before {f1_macro_before}")
        f.write(f"f1_macro_after {f1_macro_after}")
        f.write(f"accuracy_before {accuracy_before}")
        f.write(f"accuracy_after {accuracy_after}")

    precision, recall, fscore, _ = precision_recall_fscore_support(final_predictions['Right_gt'],
                                                                   final_predictions['Right_pred_before'],
                                                                   labels=labels, average=None)

    precision = pd.DataFrame(precision.reshape(1, -1), columns=labels)
    precision.to_csv(vid_name + "precision_right_before.csv")
    recall = pd.DataFrame(recall.reshape(1, -1), columns=labels)
    recall.to_csv(vid_name + "recall_right_before.csv")
    fscore = pd.DataFrame(fscore.reshape(1, -1), columns=labels)
    fscore.to_csv(vid_name + "f1_right_before.csv")
    print()
    precision, recall, fscore, _ = precision_recall_fscore_support(final_predictions['Right_gt'],
                                                                   final_predictions['Right_pred_after'],
                                                                   labels=labels, average=None)
    precision = pd.DataFrame(precision.reshape(1, -1), columns=labels)
    precision.to_csv(vid_name + "precision_right_after.csv")
    recall = pd.DataFrame(recall.reshape(1, -1), columns=labels)
    recall.to_csv(vid_name + "recall_right_after.csv")
    fscore = pd.DataFrame(fscore.reshape(1, -1), columns=labels)
    fscore.to_csv(vid_name + "f1_right_after.csv")
    precision, recall, fscore, _ = precision_recall_fscore_support(final_predictions['Left_gt'],
                                                                   final_predictions['Left_pred_before'],
                                                                   labels=labels, average=None)
    precision = pd.DataFrame(precision.reshape(1, -1), columns=labels)
    precision.to_csv(vid_name + "precision_left_before.csv")
    recall = pd.DataFrame(recall.reshape(1, -1), columns=labels)
    recall.to_csv(vid_name + "recall_left_before.csv")
    fscore = pd.DataFrame(fscore.reshape(1, -1), columns=labels)
    fscore.to_csv(vid_name + "f1_left_before.csv")
    precision, recall, fscore, _ = precision_recall_fscore_support(final_predictions['Left_gt'],
                                                                   final_predictions['Left_pred_after'],
                                                                   labels=labels, average=None)
    precision = pd.DataFrame(precision.reshape(1, -1), columns=labels)
    precision.to_csv(vid_name + "precision_left_after.csv")
    recall = pd.DataFrame(recall.reshape(1, -1), columns=labels)
    recall.to_csv(vid_name + "recall_left_after.csv")
    fscore = pd.DataFrame(fscore.reshape(1, -1), columns=labels)
    fscore.to_csv(vid_name + "f1_left_after.csv")
    after_pred += final_predictions['Left_pred_after'] + final_predictions['Right_pred_after']
    before_pred += final_predictions['Left_pred_before'] + final_predictions['Right_pred_before']
    gt += final_predictions['Left_gt'] + final_predictions['Right_gt']

precision_before, recall_before, fscore_before, _ = precision_recall_fscore_support(gt,
                                                                                    before_pred,
                                                                                    labels=labels, average=None)

precision_after, recall_after, fscore_after, _ = precision_recall_fscore_support(gt,
                                                                                 after_pred,
                                                                                 labels=labels, average=None)

f1_macro_before = f1_score(gt,
                           before_pred,
                           labels=labels,
                           average='macro')
f1_macro_after = f1_score(gt,
                          after_pred,
                          labels=labels,
                          average='macro')
accuracy_before = accuracy_score(gt,
                                 before_pred)
accuracy_after = accuracy_score(gt,
                                after_pred)

print(f"precision_before: {precision_before}")
print(f"precision_after: {precision_after}")
print(f"recall_before: {recall_before}")
print(f"recall_after: {recall_after}")
print(f"fscore_before: {fscore_before}")
print(f"fscore_after: {fscore_after}")
print()
print(f"accuracy_before: {accuracy_before}")
print(f"accuracy_after: {accuracy_after}")
