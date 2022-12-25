import argparse
import os
import warnings
import time
import cv2
import numpy as np
import pandas as pd

from src.generate_patches import CropImage
from src.model_test import AntiSpoofPredict
from src.utility import parse_model_name

warnings.filterwarnings("ignore")


def test(image, model_dir):
    model_test = AntiSpoofPredict()
    image_cropper = CropImage()

    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 2))

    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
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
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))

    label = np.argmax(prediction)
    value = prediction[0][label] / 2
    if label == 1:
        return value
    else:
        return 1 - value


def main(args):
    print("preprocessing ...")
    target = {}
    for video_name in os.listdir(args.data):
        video_path = os.path.join(args.data, video_name)
        cap = cv2.VideoCapture(video_path)
        ls = []
        c = 0
        while cap.isOpened():
            _, frame = cap.read()
            try:
                # if c % 5 == 0:
                score = test(frame, args.model_dir)
                ls.append(score)
                # c += 1
            except:
                break
        target[video_name] = sum(ls) / len(ls)
    df = pd.DataFrame(list(target.items()), columns=["fname", "liveness_score"])
    os.makedirs("results/", exist_ok=True)
    df.to_csv(
        "results/submission.csv", index=False, encoding="utf-8", float_format="%.10f"
    )
    print("output will be saved in /result/submission.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="test")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./resources/ckpt/",
        help="model_lib used to test",
    )
    parser.add_argument(
        "--data",
        type=str,
        help="image used to test",
        default="./data/",
    )
    args = parser.parse_args()

    print('Processing ... ')
    start_time = time.time()
    main(args)
    print("Save file")
    print(f"Time: {time.time() - start_time}")
