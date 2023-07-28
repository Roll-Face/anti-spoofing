import logging
import os
from logging.config import fileConfig

import cv2
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from torchvision import transforms
from tqdm import tqdm

from utils.utils import build_network, get_optimizer, read_cfg

_log_config_path = "logging.conf"
fileConfig(_log_config_path, disable_existing_loggers=False)
_logger = logging.getLogger(__name__)

import argparse

parser = argparse.ArgumentParser(description="testings")
parser.add_argument(
    "--name_dataset",
    type=str,
    default="nua",
    help="Name datasets",
)
parser.add_argument(
    "--folder_image",
    type=str,
    default="data/nuaa/images",
    help="consist images",
)

parser.add_argument("--folder_model", type=str, default="experiments/output", help="consist weight")
parser.add_argument("--file_csv", type=str, default="data/nuaa/labels.csv", help="consist weight")

args = parser.parse_args()



def test(model, folder_model, folder_image, file_csv, device, test_transform):
    _logger.info(f"datasets: {args.name_dataset}")
    for name_model in os.listdir(folder_model):
        _logger.info(name_model)
        model_path = os.path.join(folder_model, name_model)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state["state_dict"])
        model = model.to(device=device)

        data = pd.read_csv(file_csv)
        label_true = data["label"].to_list()
        label_pred = []
        list_data = {}
        for image_name in tqdm(data["name"]):
            image_path = os.path.join(folder_image, image_name)
            image = cv2.imread(image_path)
            image_transform = test_transform(img=image)
            image_transform = image_transform.to(device)
            image_transform = image_transform.unsqueeze(0)
            with torch.no_grad():
                net_depth_map, _, _, _, _, _  = network(image_transform)
            res = torch.mean(net_depth_map).item()
            # binary = torch.mean(binary).item()
            
            # res = (res + binary) / 2
            
            if res > 0.5:
                label = 1
            else:
                label = 0
            list_data[image_name] = label
            label_pred.append(label)

        precison = precision_score(
            y_true=label_true, y_pred=label_pred, average="macro"
        )
        recall = recall_score(y_true=label_true, y_pred=label_pred, average="macro")
        f1 = f1_score(y_true=label_true, y_pred=label_pred, average="macro")
        _logger.info(f"precision: {precison} - recall: {recall} - f1-score:{f1}")
        dataframe = pd.DataFrame(list_data.items(), columns=["name", "label"])
        dataframe.to_csv(f"experiments/logs/{args.name_dataset}/{name_model}_labels.csv", index=False)


if __name__ == "__main__":
    cfg = read_cfg(cfg_file="config/CDCNpp_adam_lr1e-3.yaml")
    device = torch.device("cuda:1")
    network = build_network(cfg)
    network.eval()
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(cfg['model']['input_size']),
        transforms.ToTensor(),
        transforms.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'])
    ])

    test(
        model=network,
        folder_image=args.folder_image,
        folder_model=args.folder_model,
        file_csv=args.file_csv,
        device=device,
        test_transform=test_transform,
    )
