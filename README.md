# convert-GFPGANv1.3-to-ncnn
1 Clone repository 
2 Download GFPGANv1.3.pth in root repository dir
3 Run - Export_PTH_to_PT_GFPGAN.py to convert pth to pt
4 Download pnnx from https://github.com/pnnx/pnnx and unpack to root dir of this project
5 Run pnnx 
```
pnnx.exe gfpganv1_clean_model.pt inputshape=[1,3,512,512]
```
6 Output will be gfpganv1_clean_model.ncnn.param and gfpganv1_clean_model.ncnn.bin
