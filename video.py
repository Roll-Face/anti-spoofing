import onnxruntime as rt
import torch
from src.data import transform as trans
import torch.nn.functional as F
from src.generate_patches import CropImage
from src.model_test import AntiSpoofPredict,Detection
import os 
import cv2
import numpy as np
from src.utility import parse_model_name
import time as t
from torchvision import transforms

from ultralytics import YOLO


def load_model(onnx_path):
    model = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    return model


def predict_onnx(model, img: torch.Tensor) -> np.ndarray:
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    pred_onx = model.run(
        [output_name], {input_name: (img.cpu().numpy()).astype(np.float32)}
    )[0]
    return pred_onx

class LivenessDetection:
    def __init__(self, checkpoint_path: str) -> None:
        self.CDCNpp = rt.InferenceSession(
            checkpoint_path, providers=['CPUExecutionProvider']
        )
        self.transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
            ]
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __call__(self, img):
        image_transform = self.transform(img=img)
        image_transform = image_transform.unsqueeze(0)
        image_transform = image_transform.to(self.device)
        map_x, x_concat, attention1, attention2, attention3, x_input = self.CDCNpp.run(
            ["map_x", "x_concat", "attention1", "attention2", "attention3", "x_input"],
            {"input": image_transform.detach().cpu().numpy().astype(np.float32)},
        )
        print(map_x.shape)
        score = np.mean(map_x)

        return score

def forward(img, model, device):
    test_transform = trans.Compose(
        [   
            trans.ToPILImage(),
            trans.ToTensor(),
        ]
    )
    img = test_transform(img)
    img = img.to(device)
    img = img[None, :, :]
    result = predict_onnx(model, img)

    result = F.softmax(torch.Tensor(result)).cpu().numpy()
    return result

def get_bbox_yolo(image,img_size = 640,conf_thres = 0.15,iou_thres=0.5,device=torch.device("cuda")):
    
    results = model_yolo.predict(
            source=image,
            imgsz=img_size,
            conf=conf_thres,
            iou=iou_thres,
            augment=False,
            device=device,
        )
    result = results[0].cpu().numpy()
    bboxes = []
    for box in result.boxes:
        conf = box.conf[0]

        xyxy = box.xyxy[0]
        x1 = int(xyxy[0] + 0.5)
        y1 = int(xyxy[1] + 0.5)
        x2 = int(xyxy[2] + 0.5)
        y2 = int(xyxy[3] + 0.5)

        bboxes.append([x1,y1,x2,y2])
    return bboxes

def predict(image, model_dir):
    t0 = t.time()
    image_cropper = CropImage()
    # model_test = Detection()
    # image_bbox = model_test.get_bbox(image)
    image_bboxs = get_bbox_yolo(image=image)
    print(image_bboxs)
    if len(image_bboxs) != 0:
        for image_bbox in image_bboxs:
            image_bbox[0] = max(image_bbox[0], 0)
            image_bbox[1] = max(image_bbox[1], 0)
            anti_img = image[image_bbox[1] : (image_bbox[3]), image_bbox[0] : (image_bbox[2])]
            score = livenessDetector(img=anti_img)

            prediction = np.zeros((1, 2))
            for model_name in os.listdir(model_dir):
                model_path = os.path.join(model_dir, model_name)

                h_input, w_input, _, scale = parse_model_name(model_name)
                param = {
                    "org_img": image,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }
                if scale is None:
                    param["crop"] = False
                img = image_cropper.crop(**param)
                model = load_model(model_path)
                out = forward(img, model, device=torch.device("cuda"))
                prediction += out
            
            t1 = t.time()
            fps = round(1 / (t1 - t0))
            label = np.argmax(prediction)
            value = prediction[0][label] / 2

            print(label,value,score)

            if label == 1 and score >= 0.65:
                result_text = "RealFace Score: {:.2f}".format(score)
                color = (0, 0, 255)
            else:
                result_text = "FakeFace Score: {:.2f}".format(score)
                color = (255, 0, 0)

            cv2.rectangle(
                image,
                (image_bbox[0], image_bbox[1]),
                (image_bbox[2], image_bbox[3]),
                color, 2)
            cv2.putText(
                image,
                result_text,
                (image_bbox[0], image_bbox[1] - 5),
                cv2.FONT_HERSHEY_COMPLEX, 0.6, color)
            
        cv2.putText(
            image,
            str(fps),
            (5,5),
            cv2.FONT_HERSHEY_COMPLEX, 1, (255,43,2))

    return cv2.cvtColor(image,cv2.COLOR_RGB2BGR)


def main():
    import cv2
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        image = predict(image=frame_rgb,model_dir="resources/pre-trained/fine_tune_old")
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("video: ",image)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == "__main__":
    model_yolo = YOLO("/home/namnh/Desktop/tests/models/yolov8n-face.pt")
    livenessDetector = LivenessDetection(checkpoint_path="/home/namnh/Desktop/tests/CDCN-Face-Anti-Spoofing.pytorch/88_CDCNpp_zalo.onnx")
    main()