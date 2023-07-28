import onnxruntime
import torch
from models.CDCNs import CDCNpp

device = torch.device("cuda")
def convert_ONNX(model, save_name):
    model = model.to(device)
    x = torch.randn(1, 3, 256, 256, requires_grad=True).to(device).float()
    torch.onnx.export(
        model,
        x,
        save_name,
        export_params=True,
        input_names=["input"],
        output_names=['map_x', 'x_concat', 'attention1', 'attention2', 'attention3', 'x_input'],
        dynamic_axes={
            "input": {0: "batch_size"},  # variable length axes
            "output": {0: "batch_size"},
        },
    )

def load_model(ckpt_path):
    model = CDCNpp()
    ckpt = torch.load(ckpt_path,map_location=torch.device("cuda:0"))
    model.load_state_dict(ckpt['state_dict'])
    # model.eval()
    return model

       
if __name__ == "__main__":
    # import os
    # root_dir = "Pretrained_models"
    # for model_name in os.listdir(root_dir):
    #     model_path = os.path.join(root_dir,model_name)
    #     model = load_model(ckpt_path=model_path)
    #     name_model_onnx = model_name.split(".")[0]
    #     convert_ONNX(model=model,save_name=os.path.join("pre_trained_onnx",f"{name_model_onnx}.onnx"))

    model = load_model("experiments/output/88_CDCNpp_zalo.pth")
    convert_ONNX(model,save_name="88_CDCNpp_zalo.onnx")