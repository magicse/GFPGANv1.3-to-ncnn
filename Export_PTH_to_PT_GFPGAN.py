# importing module
import sys
sys.path.append('./')

import torch
import torchvision
#import model
#import gfpgan.archs.gfpganv1_clean_arch as gfpganv1_clean_arch
from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
model_path = "GFPGANv1.3.pth"

# An instance of your model.
#model = torchvision.models.resnet18(pretrained=True)

model = GFPGANv1Clean(
    out_size=512,
    num_style_feat=512,
    channel_multiplier=2,
    decoder_load_path=None,
    fix_decoder=False,
    num_mlp=8,
    input_is_latent=True,
    different_w=True,
    narrow=1,
    sft_half=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loadnet = torch.load(model_path)
if 'params_ema' in loadnet:
    keyname = 'params_ema'
else:
    keyname = 'params'

#model.load_state_dict(loadnet[keyname], strict=True)
model.load_state_dict(loadnet[keyname], strict=False)    

# Switch the model to eval model
model.eval()
model = model.to(device)

# An example input you would normally provide to your model's forward() method.
example1 = torch.rand(1, 3, 512, 512)
#example2 = torch.rand(2, 256, 256)
#example3 = torch.rand(1, 256, 256)

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, (example1))

# Save the TorchScript model
traced_script_module.save("gfpganv1_clean_model.pt")