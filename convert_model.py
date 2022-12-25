import numpy as np
import onnxruntime as rt
import tensorrt as trt
import torch
from PIL import Image

from src.data import transform as trans
from src.model.MiniFASNet import (MiniFASNetV1, MiniFASNetV1SE,
                                      MiniFASNetV2, MiniFASNetV2SE)
from src.model_lib.MultiFTNet import MultiFTNet
from src.utility import get_kernel, parse_model_name

# -----------------------------------------------------------------------------------------------#

MODEL_MAPPING = {
    "MiniFASNetV1": MiniFASNetV1,
    "MiniFASNetV2": MiniFASNetV2,
    "MiniFASNetV1SE": MiniFASNetV1SE,
    "MiniFASNetV2SE": MiniFASNetV2SE,
}
model_type = "MiniFASNetV1SE"
device = torch.device("cuda")

model = MultiFTNet(
    model_type=MODEL_MAPPING[model_type],
    conv6_kernel=get_kernel(80, 80),
    img_channel=3,
    num_classes=2,
    training=False,
    pre_trained=None,
).to(device)

state_dict = torch.load(
    "./resources/ckpt/2.7_80x80_MiniFASNetV1SE.pth", map_location=device
)
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

# ---------------------------------------------------------------------------------------------------#


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


export_path = "/home/eco0936_namnh/CODE/zalo-challenge/resources/ckpt_onnx/1_80x80_MiniFASNetV1SE.onnx"
model = rt.InferenceSession(export_path, providers=["CUDAExecutionProvider"])


def predict_onnx(model, img: torch.Tensor) -> np.ndarray:
    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    pred_onx = model.run([output_name], {input_name: img.detach().cpu().numpy()})[0]
    return pred_onx


# ------------------------------------------------------------------------------------#
import pycuda.driver as cuda
import pycuda.autoinit
import time
import copy
import numpy as np
import os
import torch
import cv2
from pathlib import Path
cuda.init()
print('CUDA device query (PyCUDA version) \n')
import argparse

TRT_LOGGER = trt.Logger(trt.Logger.INFO)
a=(int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def GiB(val):
    return val * 1 << 30

def build_detec_engine(onnx_path, using_half, int8, engine_file  = None, dynamic_input=True, workspace_size=5,
                min_shape=(1,3,80,80), opt_shape=(1,3,80,80), max_shape=(1,3,80,80)):
    trt.init_libnvinfer_plugins(None, '')
    # initialize TensorRT engine and parse ONNX model
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(EXPLICIT_BATCH) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_batch_size = 1 # always 1 for explicit batch
        config = builder.create_builder_config()
        # allow TensorRT to use up to 1GB of GPU memory for tactic selection
        config.max_workspace_size = GiB(int(workspace_size))
        if using_half:
            config.set_flag(trt.BuilderFlag.FP16)
        if int8:
            config.set_flag(trt.BuilderFlag.INT8)
        # Load the Onnx model and parse it in order to populate the TensorRT network.
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                print ('ERROR: Failed to parse the ONNX file.')
                for error in range(parser.num_errors):
                    print (parser.get_error(error))
                return None

        if dynamic_input:
            profile = builder.create_optimization_profile();
            profile.set_shape("input", min_shape, opt_shape, max_shape)
            config.add_optimization_profile(profile)

        return builder.build_engine(network, config)

# -------------------------------------------------------------------------------------------#

if __name__ == "__main__":

    # convert_ONNX(model,"/home/eco0936_namnh/CODE/zalo-challenge/resources/ckpt_onnx/2.7_80x80_MiniFASNetV1SE.onnx")
    img = torch.randn((1,3,80,80)).to(device)
    print(rt.get_device())
    print(model)
    pred = predict_onnx(model,img)
    print(pred)

# detec_engine = build_detec_engine(onnx_path=export_path,using_half=False,int8=True)
# output_detec = os.path.join('./', "detec_trt.engine")
# with open(output_detec, "wb") as f:
#     f.write(detec_engine.serialize())
