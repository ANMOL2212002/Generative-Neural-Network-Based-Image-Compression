import os
from PIL import Image

def compress_image(image_path, output_path, quality=10):
    """Compress an image to 10% of original size"""
    file_name = os.path.basename(image_path)
    image = Image.open(image_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    
    file_name = file_name.split('.')[0] + f'_q{quality}.' + file_name.split('.')[1]
    image.save(output_path + file_name, 'JPEG', optimize=True, quality=quality)

if __name__ == '__main__':
    for q in [1,5,10,50,80,100]:
        compress_image('test.jpg', 'output/', quality=q)


