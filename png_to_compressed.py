import cv2
import os

# function to compress image using above method
def compress_image(image_path, quality=10, save_path=None):
    """Compress an image to 10% of original size"""
    image = cv2.imread(image_path)
    # file_name = file_name.split('.')[0] + f'_q{quality}.' + file_name.split('.')[1]
    cv2.imwrite(save_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

if __name__=='__main__':
    image_path = 'test.png'
    output_path = 'compressed_images/'
    compress_image(image_path, output_path, quality=10)