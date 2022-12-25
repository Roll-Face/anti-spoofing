import glob
import os
import warnings

import cv2
import pandas as pd
import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

warnings.filterwarnings("ignore")

data_folders = [
    "/home/ai/datasets/challenge/liveness/train/",
]
save_dir = "/home/ai/datasets/challenge/liveness/generate/"
os.makedirs(save_dir, exist_ok=True)

SKIP_FRAME = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MTCNN(device="cuda" if torch.cuda.is_available() else "cpu")


for idx_folder, data_dir in enumerate(data_folders):
    videos = glob.glob(os.path.join(data_dir, "videos/*.mp4"))
    save_images_dir = os.path.join(save_dir, str(idx_folder))
    os.makedirs(save_images_dir, exist_ok=True)
    face_file = open(os.path.join(save_images_dir, "face_crops.txt"), "a")

    for file in tqdm(
        videos, desc="Process {}/{}".format(idx_folder + 1, len(data_folders))
    ):
        vidcap = cv2.VideoCapture(file)
        success, frame = vidcap.read()

        count = 0
        frame_num = 0
        while success:
            torch_frame = torch.from_numpy(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).to(
                DEVICE
            )

            detect_res = model.detect(torch_frame)
            if len(detect_res) > 0 and count % SKIP_FRAME == 0:
                file_name = os.path.join(
                    save_dir,
                    str(idx_folder),
                    f"{os.path.basename(file)[:-4]}_frame_{frame_num}.jpg",
                )
                bbox = detect_res[0]

                label_org = pd.read_csv(os.path.join(data_dir, "label.csv"))
                for i in range(len(label_org)):
                    if os.path.basename(file) == label_org.loc[i, "fname"]:
                        label = label_org.loc[i, "liveness_score"]

                try:
                    # print(file_name, int(bbox[0][0]), int(bbox[0][1]), int(bbox[0][2]), int(bbox[0][3]), label)
                    face_file.writelines(
                        "%s %d %d %d %d %d\n"
                        % (
                            file_name,
                            int(bbox[0][0]),
                            int(bbox[0][1]),
                            int(bbox[0][2]),
                            int(bbox[0][3]),
                            label,
                        )
                    )
                except:
                    break
                cv2.imwrite(file_name, frame)
                frame_num += 1
            success, frame = vidcap.read()
            count += 1
            # cv2.imshow('test',frame)
            # cv2.waitKey(1)
        vidcap.release()
