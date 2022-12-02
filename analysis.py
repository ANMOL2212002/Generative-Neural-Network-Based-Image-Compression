from png_to_compressed import compress_image
from k_means_compression import k_means_compression
from skimage import io
import cv2
import os
from error_functions import *
import datetime


def analysis(og_path, compressed_folder=None, GAN_folder=None, output_folder=None):
    def check_folder(folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
    check_folder(compressed_folder)

    image_original = io.imread(og_path)
    size_original = os.path.getsize(og_path)

    # DO COMPRESSION--------------------------------
    # image_k_means
    k_means_compression(og_path, 4, compressed_folder + '/k_means.png')
    # image_jpeg_10
    compress_image(og_path, 10, compressed_folder + '/jpeg_10.jpeg')
    # image_jpeg_1
    compress_image(og_path, 1, compressed_folder + '/jpeg_1.jpeg')
    # OPEN COMPRESSED--------------------------------
    # image_k_means
    image_k_means = io.imread(compressed_folder + '/k_means.png')
    size_k_means = os.path.getsize(compressed_folder + '/k_means.png')
    # image_jpeg_10
    image_jpeg_10 = io.imread(compressed_folder + '/jpeg_10.jpeg')
    size_jpeg_10 = os.path.getsize(compressed_folder + '/jpeg_10.jpeg')

    # image_jpeg_1
    image_jpeg_1 = io.imread(compressed_folder + '/jpeg_1.jpeg')
    size_jpeg_1 = os.path.getsize(compressed_folder + '/jpeg_1.jpeg')

    # GAN_image
    image_GAN = io.imread(GAN_folder + '/' + os.listdir(GAN_folder)[-1])
    size_GAN = os.path.getsize(GAN_folder + '/../' + 'SSIM.pt')

    IMAGES = [image_original, image_k_means,
              image_jpeg_10, image_jpeg_1, image_GAN]
    IMAGE_NAMES = ['PNG', 'K-Means', 'JPEG 10%', 'JPEG 1%', 'GAN' + ' - after ' +
                   os.listdir(GAN_folder)[-1].split('_')[-1].split('.')[0] + ' epochs']
    IMAGE_SIZES = [size_original, size_k_means,
                   size_jpeg_10, size_jpeg_1, size_GAN]

    # COMPARE--------------------------------
    check_folder(output_folder)
    # make csv file
    with open(output_folder + '/analysis.csv', 'w') as f:
        f.write('scheme, BPP, CR, PSNR, MSE, SSIM\n')
        for image, name, size in zip(IMAGES, IMAGE_NAMES, IMAGE_SIZES):
            bpp, cr, psnr, mse, ssim = error_functions(
                image_original, image, size_original, size)
            f.write(f'{name}, {bpp}, {cr}, {psnr}, {mse}, {ssim}\n')


if __name__ == "__main__":
    path = "/home/aryan/sem-5/smai/project/57/check"
    analysis(path+'/original/image_.png', compressed_folder=path +
             '/compressed/', GAN_folder=path + '/GAN/', output_folder=path + '/analysis/')
