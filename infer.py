import logging
import os
from logging.config import fileConfig

import cv2
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torchvision import transforms
from tqdm import tqdm

from utils.utils import build_network, get_optimizer, read_cfg
from detect_face import FaceDetection

from ultralytics import YOLO

yolo = YOLO("/home/namnh/Desktop/tests/models/yolov8n-face.pt")

def detect_face_yolov8(processed: np.ndarray):
    results = yolo.predict(
        source=processed,
        imgsz=640,
        conf=0.6,
        iou=0.8,
        augment=False,
        device=device,
    )
    result = results[0].cpu().numpy()
    detected_boxes = []
    for box in result.boxes:
        xyxy = box.xyxy[0]

        x1 = int(xyxy[0] + 0.5)
        y1 = int(xyxy[1] + 0.5)
        x2 = int(xyxy[2] + 0.5)
        y2 = int(xyxy[3] + 0.5)

        detected_boxes.append([x1, y1, x2, y2])

    return detected_boxes

cfg = read_cfg(cfg_file="config/CDCNpp_adam_lr1e-3.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
network = build_network(cfg)
# cung ok: 88,85
# cung vua vua: 98,76,79
ckpt = torch.load("experiments/output/88_CDCNpp_zalo.pth",map_location=device)
state_dict = ckpt['state_dict']
network.load_state_dict(state_dict=state_dict)
network.to(device)
network.eval()
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(cfg['model']['input_size']),
    transforms.ToTensor(),
    transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
])
cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    if ret:
        img_det = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bboxes = detect_face_yolov8(processed=img_det)
        for box in bboxes:
            box[0] = max(box[0], 0)
            box[1] = max(box[1], 0)

            anti_img = img_det[box[1]:(box[3]), box[0]:(box[2])]
            image = test_transform(anti_img)
            image = image.unsqueeze(0)
            image = image.to(device)
            with torch.no_grad():
                net_depth_map, _, _, _, _, _  = network(image)
            print(net_depth_map.shape)
            new = net_depth_map.cpu().numpy().transpose(2,1,0)
            
            res = torch.mean(net_depth_map).item()
            if res < 0.65:
                cv2.putText(img, f'Fake:{res}', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1)
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0,0,255), 1)
            else:
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255,0,0), 1)
                cv2.putText(img, f'Real:{res}', (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
        cv2.imshow("depth map",new)
        img = cv2.resize(img,(480,640))
        cv2.imshow('Anti spoofing', img)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break



