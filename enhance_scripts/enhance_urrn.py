import torch, os, glob
import numpy as np
from PIL import Image
import torchvision
import model

SRC = "/home/anushcka/datasets/euroc/synthesized/MH_01/severe"
OUT = "/home/anushcka/datasets/euroc/urrnet_enhanced/MH_01/severe"
os.makedirs(OUT, exist_ok=True)

net = model.UVRE().cuda()
net.load_state_dict(torch.load("model/Epoch99.pth"))
net.eval()

files = sorted(glob.glob(os.path.join(SRC, "*.png")))[:20]

with torch.no_grad():
    for f in files:
        img = Image.open(f).convert("RGB")
        x = np.asarray(img) / 255.0
        x = torch.from_numpy(x).float().permute(2, 0, 1).unsqueeze(0).cuda()
        _, enhanced, _, _ = net(x)
        torchvision.utils.save_image(enhanced, os.path.join(OUT, os.path.basename(f)))
        print("done:", os.path.basename(f))

print("finished", len(files), "frames :", OUT)
