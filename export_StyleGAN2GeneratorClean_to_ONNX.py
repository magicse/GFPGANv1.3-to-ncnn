# -*- coding: utf-8 -*-
import cv2
from basicsr.utils import img2tensor, tensor2img
from torchvision.transforms.functional import normalize

import torch
import onnx
import onnxruntime as ort
from onnxsim import simplify


#from colorizers import eccv16, siggraph17
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
from gfpgan.archs.stylegan2_clean_arch import StyleGAN2GeneratorClean

def convert_static_GFPGANv1Clean_1_3_onnx():
    onnx_path = "./pretrained/StyleGAN2GeneratorClean.onnx"
    sim_onnx_path = "./pretrained/StyleGAN2GeneratorClean_sim.onnx"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'GFPGANv1.3.pth'

    inference_model = StyleGAN2GeneratorClean(
        out_size=8,
        num_style_feat=512,
        channel_multiplier=2,
        num_mlp=8,
        narrow=1).to(device)
    
    loadnet = torch.load(model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    #inference_model.load_state_dict(loadnet[keyname], strict=True)
    inference_model.load_state_dict(loadnet[keyname], strict=False)
    
    
    inference_model = inference_model.eval()
    style_code = torch.rand((1, 512), dtype=torch.float32)
    conditions = torch.randn(1, 1, 1)    

    
    torch.onnx.export(inference_model,  # model being run
                      ([style_code], conditions),  # model input (or a tuple for multiple inputs)
                      onnx_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=True,
                      input_names=['input'],  # the model's input names
                      output_names=['out_ab']  # the model's output names
                      )

    print("export StyleGAN2GeneratorClean onnx done.")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    model_simp, check = simplify(onnx_model, check_n=3)
    onnx.save(model_simp, sim_onnx_path)
    print("export StyleGAN2GeneratorClean onnx sim done.")

    # test onnxruntime
    ort_session = ort.InferenceSession(sim_onnx_path)

    x_numpy = x.cpu().numpy()

    out_ab = ort_session.run(['out_ab'], input_feed={"input": x_numpy})

    print("siggraph17 out_ab[0].shape: ", out_ab[0].shape)


if __name__ == "__main__":
    #convert_static_GFPGANv1Clean_1_3_onnx()
    convert_static_GFPGANv1Clean_1_3_onnx()

    """cmd
    PYTHONPATH=. python3 ./export_onnx.py
    """