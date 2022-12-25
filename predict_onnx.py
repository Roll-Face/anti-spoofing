import os
import time 
import cv2
import numpy as np
import onnxruntime as rt
import torch
import torch.nn.functional as F
import pandas as pd 
import argparse
from src.data import transform as trans
from src.generate_patches import CropImage
from src.model_test import Detection
from src.utility import parse_model_name
import warnings
warnings.filterwarnings('ignore')


def load_model(onnx_path):
    model = rt.InferenceSession(onnx_path, providers=['CUDAExecutionProvider'])
    return model

def predict_onnx(model, img: torch.Tensor) -> np.ndarray:
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    pred_onx = model.run([output_name], {input_name: (img.cpu().numpy()).astype(np.float32)})[0]
    return pred_onx

def forward(img, model, device):
    test_transform = trans.Compose(
        [
            trans.ToTensor(),
        ]
    )
    img = test_transform(img)
    img = img.to(device)
    img = img[None,:,:]
    result = predict_onnx(model,img)
    
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
        prediction += forward(img, model,device=torch.device('cuda'))
    label = np.argmax(prediction)
    score = prediction[0][label] / 2
    if label == 1:
        return score
    else:
        return 1 - score

def main(video_dir,model_dir):
    target = {}
    try:
        for video_name in os.listdir(video_dir):
            ls = []
            count  = 0
            cap = cv2.VideoCapture(os.path.join(video_dir,video_name))
            while True:
                _,frame = cap.read()
                try:
                    if count % 5 == 0:
                        score = predict(frame,model_dir)
                        ls.append(score)
                    count += 1
                except:
                    break
            target[video_name] = sum(ls)/len(ls)
        dataframe = pd.DataFrame(list(target.items()),columns=['fname','liveness_score'])
        os.makedirs('/result',exist_ok=True)
        dataframe.to_csv("/result/submission.csv",index=False)
    except Exception as e:
        print(f'Error {e}')
        
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
    print('Pre-Processing ... ')
    start_time = time.time()
    main(video_dir=args.data,model_dir=args.model_dir)
    print("output will be saved in /result/submission.csv")
    print(f"Time: {time.time() - start_time}")