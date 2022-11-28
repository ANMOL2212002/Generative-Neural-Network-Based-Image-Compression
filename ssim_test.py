import pytorch_ssim
import torch
from torch.autograd import Variable
import numpy as np
import cv2
from piqa import SSIM
from skimage.metrics import structural_similarity as ssim


class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)


criterion = SSIMLoss().cuda()  # .cuda() if you need GPU support

...

img1 = torch.rand(1, 3, 256, 256).cuda()
img2 = torch.rand(1, 3, 256, 256).cuda()
if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()

ssim_noise = ssim(img1, img2,
                  data_range=img1.max() - img2.min())
loss = pytorch_ssim.ssim(img1, img2)
