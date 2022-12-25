import os
import warnings

import torch
import torch.nn.functional as F
from torch import optim
from torch.nn import CrossEntropyLoss, MSELoss
from tqdm import tqdm

from src.data.dataset_loader import get_train_loader, get_val_loader
from src.method_evaluate import get_equal_error_rate, get_tp_fp_rates
from src.model.MiniFASNet import (MiniFASNetV1, MiniFASNetV1SE, MiniFASNetV2,
                                  MiniFASNetV2SE)
from src.model_lib.MultiFTNet import MultiFTNet

warnings.filterwarnings("ignore")

MODEL_MAPPING = {
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}
import wandb


class TrainMain:
    def __init__(self, conf):
        self.conf = conf
        self.board_loss_every = conf.board_loss_every
        self.save_every = conf.save_every
        self.step = 0
        self.start_epoch = 0
        self.train_loader = get_train_loader(self.conf)
        self.valid_loader = get_val_loader(self.conf)

        wandb.init(
            project="zalo-challenge-liveness-detection",
            entity="namnguyen3",
            config={
                "architecture": self.conf.model_type,
                "epochs": self.conf.epochs,
                "batch_size": self.conf.batch_size,
                "optimizer": self.conf.optimizer_type,
                "schedule_type": self.conf.schedule_type,
                "label_list": os.path.basename(self.conf.label_list),
                "lr": self.conf.lr,
                "input_size": self.conf.input_size,
                "kernel_size": self.conf.kernel_size,
                "pre_trained": os.path.basename(self.conf.pre_trained),
            },
        )

    def train_model(self):
        self._init_model_param()
        self._train_stage()

    def _init_model_param(self):
        self.cls_criterion = CrossEntropyLoss()
        self.ft_criterion = MSELoss()
        self.model = self._define_network()

        if self.conf.optimizer_type == "sgd":
            self.optimizer = optim.SGD(
                self.model.module.parameters(),
                lr=self.conf.lr,
                weight_decay=5e-4,
                momentum=self.conf.momentum,
            )
        elif self.conf.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                self.model.module.parameters(),
                lr=self.conf.lr,
                amsgrad=True,
                weight_decay=1.0e-5,
            )
        else:
            self.optimizer = None

        if self.conf.schedule_type == "MultiStepLR":
            self.schedule_lr = optim.lr_scheduler.MultiStepLR(
                self.optimizer, self.conf.milestones, self.conf.gamma, -1
            )
        elif self.conf.schedule_type == "ReduceLROnPlateau":
            self.schedule_lr = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=0.01, verbose=True, patience=5
            )
        else:
            self.schedule_lr = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=10, eta_min=0.00000001, verbose=True
            )

    def _define_network(self):
        param = {
            "model_type": MODEL_MAPPING[self.conf.model_type],
            "num_classes": self.conf.num_classes,
            "img_channel": self.conf.input_channel,
            "embedding_size": self.conf.embedding_size,
            "conv6_kernel": self.conf.kernel_size,
            "training": True,
            "pre_trained": self.conf.pre_trained,
        }

        model = MultiFTNet(**param).to(self.conf.device)
        model = torch.nn.DataParallel(model, self.conf.devices)
        model.to(self.conf.device)
        return model

    def _train_stage(self):
        for e in range(self.start_epoch, self.conf.epochs):
            print(f"--------Epoch----------: {e}")
            (
                loss_train,
                loss_cls,
                loss_fea,
                acc,
                eer_train,
                lr,
            ) = self._train_batch_data()
            (
                loss_val,
                eer_val,
                acc_val,
                total_pred,
                total_label,
            ) = self._valid_batch_data()

            wandb.log(
                {
                    "loss-train": loss_train,
                    "loss_cls": loss_cls,
                    "loss_fea": loss_fea,
                    "acc-train": acc,
                    "eer_train": eer_train,
                    "loss-valid": loss_val,
                    "acc-val": acc_val,
                    "eer_val": eer_val,
                    "learning-rate": lr,
                    "valid_confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=total_label.detach().cpu().numpy(),
                        preds=total_pred.detach().cpu().numpy(),
                        class_names=["fake", "real"],
                    ),
                }
            )

            torch.save(
                self.model.state_dict(),
                self.conf.model_path + "/" + self.conf.name_ckpt + f"_{e}.pth",
            )
            artifact = wandb.Artifact("model", type="model")
            artifact.add_file(
                self.conf.model_path + "/" + self.conf.name_ckpt + f"_{e}.pth"
            )
            wandb.log_artifact(artifact)
            wandb.join()

        if isinstance(self.conf.schedule_type, optim.lr_scheduler.ReduceLROnPlateau):
            self.schedule_lr.step(loss_train)
        else:
            self.schedule_lr.step()

    def _train_batch_data(self):
        self.model.train()
        loss_sum = 0
        loss_cls_sum = 0
        loss_fea_sum = 0
        acc_sum = 0
        eer_sum = 0

        for imgs, ft_sample, target in tqdm(iter(self.train_loader)):
            total_pred = torch.Tensor().to(self.conf.device)
            total_label = torch.Tensor().to(self.conf.device)

            self.optimizer.zero_grad()
            target = target.to(self.conf.device)
            embeddings, feature_map = self.model.forward(imgs.to(self.conf.device))

            loss_cls = self.cls_criterion(embeddings, target)
            loss_fea = self.ft_criterion(feature_map, ft_sample.to(self.conf.device))
            loss = 0.5 * loss_cls + 0.5 * loss_fea

            loss_sum += loss
            loss_cls_sum += loss_cls
            loss_fea_sum += loss_fea

            loss.backward()
            self.optimizer.step()
            lr = self.optimizer.param_groups[-1]["lr"]
            pred = F.softmax(embeddings, dim=1)
            pred_argmax = torch.argmax(pred, dim=1)
            total_pred = torch.cat(
                (total_pred.to(torch.int8), pred_argmax.to(torch.int8))
            )
            total_label = torch.cat((total_label.to(torch.int8), target.to(torch.int8)))

            acc = self._get_accuracy(embeddings, target)[0]
            acc_sum += acc

            fpr, tpr, threshold = get_tp_fp_rates(
                total_label.detach().cpu().numpy(), total_pred.detach().cpu().numpy()
            )

            eer = get_equal_error_rate(tpr=tpr, fpr=fpr)
            eer_sum += eer

        return (
            loss_sum / len(self.train_loader),
            loss_cls_sum / len(self.train_loader),
            loss_fea_sum / len(self.train_loader),
            acc_sum / len(self.train_loader),
            eer_sum / len(self.train_loader),
            lr,
        )

    def _valid_batch_data(self):
        self.model.eval()
        loss = 0
        eer_sum = 0
        acc_sum = 0
        total_pred = torch.Tensor().to(self.conf.device)
        total_label = torch.Tensor().to(self.conf.device)
        for imgs, _, target in tqdm(iter(self.valid_loader)):

            with torch.no_grad():
                target = target.to(self.conf.device)
                embeddings = self.model.forward(imgs.to(self.conf.device))
                loss_cls = self.cls_criterion(embeddings, target)
                loss += loss_cls

            pred = F.softmax(embeddings, dim=1)
            pred_argmax = torch.argmax(pred, dim=1)

            total_pred = torch.cat(
                (total_pred.to(torch.int8), pred_argmax.to(torch.int8))
            )
            total_label = torch.cat((total_label.to(torch.int8), target.to(torch.int8)))

            acc = self._get_accuracy(embeddings, target)[0]
            acc_sum += acc

            fpr, tpr, threshold = get_tp_fp_rates(
                total_label.detach().cpu().numpy(), total_pred.detach().cpu().numpy()
            )
            eer = get_equal_error_rate(tpr=tpr, fpr=fpr)
            eer_sum += eer
        return (
            loss / len(self.valid_loader),
            eer_sum / len(self.valid_loader),
            acc_sum / len(self.valid_loader),
            total_pred,
            total_label,
        )

    def _get_accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        ret = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(dim=0, keepdim=True)
            ret.append(correct_k.mul_(1.0 / batch_size))
        return ret

    def _save_state(self, time_stamp, extra=None):
        save_path = self.conf.model_path
        torch.save(
            self.model.state_dict(),
            save_path
            + "/"
            + ("{}_{}_model_iter-{}.pth".format(time_stamp, extra, self.step)),
        )
