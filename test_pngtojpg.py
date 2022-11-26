from png_to_compressed import compress_image
from k_means_compression import k_means_compression
from skimage import io
import cv2
import os
from error_functions import *


def analysis(og_path, compressed_folder=None, GAN_folder=None, output_folder=None):
    # if not folder create folder
    def check_folder(folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    check_folder(compressed_folder)

    image_original_io = io.imread(og_path)
    # image_original_cv2 = cv2.imread(og_path)

    # DO COMPRESSION--------------------------------
    # image_k_means
    k_means_compression(og_path, 4, compressed_folder + '/k_means.png')
    # image_jpeg_10
    compress_image(og_path, 10, compressed_folder + '/jpeg_10.jpeg')
    # image_jpeg_1
    compress_image(og_path, 1, compressed_folder + '/jpeg_1.jpeg')

    print("hey")

    # OPEN COMPRESSED--------------------------------
    # image_k_means
    image_k_means = io.imread(compressed_folder + '/k_means.png')
    # image_jpeg_10
    image_jpeg_10 = io.imread(compressed_folder + '/jpeg_10.jpeg')
    # image_jpeg_1
    image_jpeg_1 = io.imread(compressed_folder + '/jpeg_1.jpeg')
    # GAN_image
    image_GAN = io.imread(GAN_folder + '/' + os.listdir(GAN_folder)[-1])

    IMAGES = [image_original_io, image_k_means, image_jpeg_10, image_jpeg_1, image_GAN]

    # COMPARE--------------------------------
    check_folder(output_folder)
    # make csv file
    with open(output_folder + '/analysis.csv', 'w') as f:
        f.write('scheme, BPP, CR, PSNR, MSE, SSIM\n')
        for image in IMAGES:
            bpp, cr, psnr, mse, ssim = error_functions(image_original_io, image)
            f.write(f'{image}, {bpp}, {cr}, {psnr}, {mse}, {ssim}\n')

if _name_ == "_main_":
    path = "D:/Documents/IIIT hyd/SMAI/Project/SRC/57"
    analysis('./SRC/57/original/image_.png', compressed_folder = path + '/compressed/', GAN_folder = path + '/GAN/', output_folder= path + '/analysis/')