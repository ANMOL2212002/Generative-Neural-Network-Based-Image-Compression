#  ssim metric for two images using skimage

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import io
from skimage import img_as_float
from skimage import img_as_ubyte
import numpy as np
from skimage.metrics import mean_squared_error as mse
from PIL import Image
def RGBA2RGB(image):
    """
    Converts an 4 channel RGBA image to 3 channel RGB image
    :param image: Image to be converted to RGB
    :return: RGB image
    """

    if image.shape[-1] == 3:
        return image

    rgba_image = Image.fromarray(image)
    rgba_image.load()
    rgb_image = Image.new("RGB", rgba_image.size, (255, 255, 255))
    rgb_image.paste(rgba_image, mask=rgba_image.split()[3])

    return np.array(rgb_image)

def ssim_metric(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return ssim(img1, img2, data_range=img1.max() - img1.min(), channel_axis=2)


def psnr_metric(img1, img2):
    if mse_metric(img1, img2) == 0:
        return 100
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return psnr(img1, img2, data_range=img1.max() - img1.min())


def bpp_metric(img, img_size):
    return img_size * 8 / (img.shape[0] * img.shape[1])


def cr(img1, img2):
    bpp1 = bpp_metric(img1)
    bpp2 = bpp_metric(img2)
    return bpp1/bpp2


def mse_metric(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return mse(img1, img2)

def error_functions(Img1, Img2, Img1_size, Img2_size):
    img1 = RGBA2RGB(Img1)
    img2 = RGBA2RGB(Img2)
    #get size of image

    bpp = bpp_metric(img2, Img2_size)
    cr = bpp_metric(img1, Img1_size)/bpp
    psnr = psnr_metric(img1, img2)
    mse = mse_metric(img1, img2)
    ssim = ssim_metric(img1, img2)
    return bpp, cr, psnr, mse, ssim

if __name__ == "__main__":
    Img1 = io.imread('test.jpg')
    Img2 = io.imread('test_q1.jpg')
    img1 = RGBA2RGB(Img1)
    img2 = RGBA2RGB(Img2)
    
