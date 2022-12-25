import argparse
import os
import time
import warnings

import cv2
import numpy as np
import onnxruntime as rt
import pandas as pd
import torch
import torch.nn.functional as F

from src.data import transform as trans
from src.generate_patches import CropImage
from src.model_test import Detection
from src.utility import parse_model_name

warnings.filterwarnings("ignore")


def load_model(onnx_path):
    model = rt.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
    return model


def predict_onnx(model, img: torch.Tensor) -> np.ndarray:
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    pred_onx = model.run(
        [output_name], {input_name: (img.cpu().numpy()).astype(np.float32)}
    )[0]
    return pred_onx


def forward(img, model, device):
    test_transform = trans.Compose(
        [
            trans.ToTensor(),
        ]
    )
    img = test_transform(img)
    img = img.to(device)
    img = img[None, :, :]
    result = predict_onnx(model, img)

    result = F.softmax(torch.Tensor(result)).cpu().numpy()
    return result


def predict(image, model_dir):
    image_cropper = CropImage()
    model_test = Detection()
    image_bbox = model_test.get_bbox(image)
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
        prediction += forward(img, model, device=torch.device("cuda"))
    label = np.argmax(prediction)
    score = prediction[0][label] / 2
    if label == 1:
        return label, image_bbox, score
    else:
        return label, image_bbox, (1 - score)


def main(args):
    cap = cv2.VideoCapture(args.input_data)
    while True:
        _, frame = cap.read()
        try:
            label, image_bbox, score = predict(frame, args.model_dir)
            if label == 1:
                cv2.rectangle(
                    frame,
                    (image_bbox[0], image_bbox[1]),
                    (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                    (255, 0, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Real face",
                    (image_bbox[0], image_bbox[1] - 5),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5 * frame.shape[0] / 1024,
                    (255, 0, 0),
                )
            else:
                cv2.rectangle(
                    frame,
                    (image_bbox[0], image_bbox[1]),
                    (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    "Fake face",
                    (image_bbox[0], image_bbox[1] - 5),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5 * frame.shape[0] / 1024,
                    (0, 0, 255),
                )
        except:
            break
        cv2.imshow("video", frame)
        if cv2.waitKey(0) & 0xFF == ord("q"):
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/ckpt/",
        help="model_lib used to test",
    )
    parser.add_argument(
        "--input_data",
        type=str,
        help="image used to test",
    )
    args = parser.parse_args()
    start_time = time.time()
    print("Runing ...")
    main(args)
    print(f"Time: {time.time() - start_time}")
