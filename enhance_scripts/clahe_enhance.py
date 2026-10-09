import cv2
import os
import glob

SRC="/home/anushcka/datasets/euroc/synthesized/MH_01/severe"
OUT="/home/anushcka/datasets/euroc/clahe_enhanced/MH_01/severe"
os.makedirs(OUT, exist_ok=True)

clahe= cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

files= sorted(glob.glob(os.path.join(SRC,"*.png")))[:20]

for f in files:
    img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
    enhanced = clahe.apply(img)
    cv2.imwrite(os.path.join(OUT, os.path.basename(f)), enhanced)
    print("done:", os.path.basename(f))

print("finished",len(files), "frames :", OUT)
