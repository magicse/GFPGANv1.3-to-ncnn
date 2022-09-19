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


def convert_static_GFPGANv1Clean_1_3_onnx():
    onnx_path = "./pretrained/GFPGANv1.3.onnx"
    sim_onnx_path = "./pretrained/GFPGANv1.3_sim.onnx"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_path = 'GFPGANv1.3.pth'

    inference_model = GFPGANv1Clean(
        out_size=512,
        num_style_feat=512,
        channel_multiplier=2,
        decoder_load_path=None,
        fix_decoder=False,
        num_mlp=8,
        input_is_latent=True,
        different_w=True,
        narrow=1,
        sft_half=True).to(device)
    
    loadnet = torch.load(model_path)
    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'
    #inference_model.load_state_dict(loadnet[keyname], strict=True)
    inference_model.load_state_dict(loadnet[keyname], strict=False)
    
    
    inference_model = inference_model.eval()
    
    img_path = '1.jpg'
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(input_img, (512, 512))
    cropped_face_t = img2tensor(img / 255., bgr2rgb=True, float32=True)
    normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
    
    mat1 = torch.randn(3, 512, 512).cpu()  # moving the tensor to cpu
    mat1 = mat1.unsqueeze(0).to(device)
    return_rgb=False
    
    torch.onnx.export(inference_model,  # model being run
                      (cropped_face_t, return_rgb),  # model input (or a tuple for multiple inputs)
                      onnx_path,  # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=11,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      verbose=True,
                      input_names=['input'],  # the model's input names
                      output_names=['out_ab']  # the model's output names
                      )

    print("export GFPGANv1.3 onnx done.")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)

    model_simp, check = simplify(onnx_model, check_n=3)
    onnx.save(model_simp, sim_onnx_path)
    print("export GFPGANv1.3 onnx sim done.")

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
