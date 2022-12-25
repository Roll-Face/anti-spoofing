import torch
import torch.nn.functional as F
from torch import nn

from src.model_lib.MiniFASNet import (MiniFASNetV1, MiniFASNetV1SE,
                                      MiniFASNetV2, MiniFASNetV2SE)
from src.utility import set_parameter_requires_grad


class FTGenerator(nn.Module):
    def __init__(self, in_channels=48, out_channels=1):
        super(FTGenerator, self).__init__()

        self.ft = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.ft(x)


class MultiFTNet(nn.Module):
    def __init__(
        self,
        model_type=MiniFASNetV2,
        img_channel=3,
        num_classes=3,
        embedding_size=128,
        conv6_kernel=(5, 5),
        training=True,
        pre_trained=None,
    ):
        super(MultiFTNet, self).__init__()
        self.training = training
        self.img_channel = img_channel
        self.num_classes = num_classes
        self.model = model_type(
            embedding_size=embedding_size,
            conv6_kernel=conv6_kernel,
            num_classes=num_classes,
            img_channel=img_channel,
        )
        self.FTGenerator = FTGenerator(in_channels=128)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if pre_trained is not None:
            state_dict = torch.load(pre_trained, map_location=device)
            keys = iter(state_dict)
            first_layer_name = keys.__next__()
            if first_layer_name.find("module.") >= 0:
                from collections import OrderedDict

                new_state_dict = OrderedDict()
                for key, value in state_dict.items():
                    name_key = key[7:]
                    new_state_dict[name_key] = value
                self.model.load_state_dict(new_state_dict)
        else:
            self._initialize_weights()
        # set_parameter_requires_grad(self.model, feature_extracting=True)
        self.model.prob = nn.Linear(
            in_features=self.model.bn.num_features, out_features=2, bias=False
        )
        self.model.to(device)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.conv2_dw(x)
        x = self.model.conv_23(x)
        x = self.model.conv_3(x)
        x = self.model.conv_34(x)
        x = self.model.conv_4(x)
        x1 = self.model.conv_45(x)
        x1 = self.model.conv_5(x1)
        x1 = self.model.conv_6_sep(x1)
        x1 = self.model.conv_6_dw(x1)
        x1 = self.model.conv_6_flatten(x1)
        x1 = self.model.linear(x1)
        x1 = self.model.bn(x1)
        x1 = self.model.drop(x1)
        cls = self.model.prob(x1)

        if self.training is True:
            ft = self.FTGenerator(x)
            return cls, ft
        else:
            return cls


if __name__ == "__main__":
    import torch

    def get_kernel(height, width):
        kernel_size = ((height + 15) // 16, (width + 15) // 16)
        return kernel_size

    ckpt_path = "/home/ai/challenge/Silent-Face-Anti-Spoofing/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth"
    model = MultiFTNet(
        img_channel=3,
        conv6_kernel=get_kernel(80, 80),
        num_classes=3,
        training=False,
        pre_trained=ckpt_path,
    )
    model = torch.nn.DataParallel(model).cuda()
    img = torch.randn((4, 3, 80, 80)).cuda()
    out = model(img)
    print(out)
