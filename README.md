# Convert GFPGANv1.3 to ncnn
1 Clone repository<br>
2 Download GFPGANv1.3.pth in root dir of this project<br>
3 Run - Export_PTH_to_PT_GFPGAN.py to convert pth to pt<br>
4 Download pnnx from https://github.com/pnnx/pnnx and unpack to root dir of this project<br>
5 Run pnnx 
```
pnnx.exe gfpganv1_clean_model.pt inputshape=[1,3,512,512]
```
6 Output will be gfpganv1_clean_model.ncnn.param and gfpganv1_clean_model.ncnn.bin<br>

# Convert GFPGANv1.3 to onnx
1 Clone repository<br>
2 Download GFPGANv1.3.pth in root dir of this project<br>
3 Create folder "pretrained" in root dir of project<br>
4 Put any jpg file 512x512 in root dir with name 1.jpg<br>
4 Run - Export_GFPGANv1_3_to_ONNX.py to convert pth to ONNX<br>
