from enlighten_inference import EnlightenOnnxModel
import cv2, os, glob

SRC= "/home/anushcka/datasets/euroc/synthesized/MH_01/severe"
OUT= "/home/anushcka/datasets/euroc/enlighten_enhanced/MH_01/severe"
os.makedirs(OUT, exist_ok=True)

model= EnlightenOnnxModel(providers=["CPUExecutionProvider"])

files= sorted(glob.glob(os.path.join(SRC,"*.png")))[:20]

for f in files:
    img= cv2.imread(f) #3 channel BGR for EnlightenGAN
    out= model.predict(img)
    cv2.imwrite(os.path.join(OUT, os.path.basename(f)), out)
    print("done:", os.path.basename(f))

print("finished", len(files), "frames:", OUT)
