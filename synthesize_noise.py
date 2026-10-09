import cv2
import numpy as np
import os


LEVELS = {
    "normal": {"alpha": 1.0, "sigma": 0, "blur": 0},
    "mild": {"alpha": 0.5, "sigma": 0, "blur": 0},
    "severe": {"alpha": 0.3, "sigma": 10, "blur": 0},
    "extreme": {"alpha": 0.1, "sigma": 20, "blur": 15},
}

def degrade(img, alpha, sigma, blur):
    gamma = 1.0/alpha
    out = np.power(img/255.0, gamma) * 255
    if sigma > 0:
        out = out + np.random.normal(0, sigma, out.shape)
    out = np.clip(out, 0, 255).astype(np.uint8)
    if blur > 0:
        k = np.zeros((blur, blur), np.float32)
        k[blur //2, :] = 1.0/blur
        out = cv2.filter2D(out, -1, k)
    return out 

src_dir = "/home/anushcka/datasets/euroc/MH_01_easy/mav0/cam0/data/"
out_dir = "/home/anushcka/datasets/euroc/synthesized/MH_01"
# N = 150

frames = sorted(os.listdir(src_dir))

for level, p in LEVELS.items():
    os.makedirs(os.path.join(out_dir, level), exist_ok=True)

for fname in frames:
    img = cv2.imread(os.path.join(src_dir, fname), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("skip:", fname)
        continue
    for level, p in LEVELS.items():
        degraded = degrade(img, p["alpha"], p["sigma"], p["blur"])
        cv2.imwrite(os.path.join(out_dir, level, fname), degraded)
print("done:", len(frames), "frames x", len(LEVELS), "levels")

# img = cv2.imread("/home/anushcka/datasets/euroc/MH_01_easy/mav0/cam0/data/1403636579763555584.png", cv2.IMREAD_GRAYSCALE)
# if img is not None:
#     print("Loaded: ", img, img.shape)

# alpha = 0.1 # severe from LDFE
# gamma = 1.0/alpha
# dark = np.power(img/255.0, gamma) * 255 # normalize 0-1 then scale back
# dark = dark + np.random.normal(0, 20, dark.shape) # gaussian
# # After adding random noise some pixels can be out of 0-255 range
# dark = np.clip(dark, 0, 255).astype(np.uint8)

# # motion blur
# length = 15
# kernel = np.zeros((length, length), np.float32) # start with all 0s grid
# kernel[length // 2, :] = 1.0/length # fill the middle row with equal weights
# dark = cv2.filter2D(dark, -1, kernel)

# cv2.imwrite("/home/anushcka/datasets/euroc/synthesized/test_extreme.png", dark)
# print("saved test severe png")

