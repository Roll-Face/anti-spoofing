import torch
from src.model.MiniFASNet import (MiniFASNetV1, MiniFASNetV1SE, MiniFASNetV2,
                                  MiniFASNetV2SE)
from src.model.MultiFTNet import MultiFTNet
from src.utility import get_kernel


MODEL_MAPPING = {
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}


def load_model(model_type,model_path,device):
    model = MultiFTNet(
        model_type=MODEL_MAPPING[model_type],
        conv6_kernel=get_kernel(80, 80),
        img_channel=3,
        num_classes=2,
        training=False,
        pre_trained=None,
    ).to(device)

    state_dict = torch.load(model_path, map_location=device)
    keys = iter(state_dict)
    first_layer_name = keys.__next__()
    if first_layer_name.find("module.") >= 0:
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for key, value in state_dict.items():
            name_key = key[7:]
            new_state_dict[name_key] = value
        model.load_state_dict(new_state_dict)
    model.eval()

    return model


def convert_ONNX(model, save_name):
    x = torch.randn(1, 3, 80, 80, requires_grad=True).to(device)
    torch.onnx.export(
        model,
        x,
        save_name,
        export_params=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    model_type = "MiniFASNetV2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "/home/namnh/Desktop/tests/anti-spoofing/fine_tuning/2.7_80x80_MiniFASNetV2_460.pth"
    model = load_model(model_type=model_type,model_path=model_path,device=device)
    convert_ONNX(model,"/home/namnh/Desktop/tests/anti-spoofing/fine_tuning/onnx/2.7_80x80_MiniFASNetV2.onnx")