
# Download a pre-trained model from the model zoo usign tensorflow
import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
# import Image
from PIL import Image, ImageDraw
# import tensorflow as tf
import os
import torch.nn as nn
# import utils


def save_image(images, save_path, mode, iteration=None):

    PATH = f"{save_path}/{mode}"
    # os.makedirs(PATH, exist_ok=True)

    grid = torchvision.utils.make_grid(images.clamp(
        min=-1, max=1), scale_each=True, normalize=True)
    grid_image = grid.permute(1, 2, 0).cpu().numpy()
    plt.imshow(grid_image)

    if iteration:
        plt.imsave(f"{PATH}/image_{iteration}.png", grid_image)
    else:
        plt.imsave(f"{PATH}/original_image.png", grid_image)


def compressor(model, image, save_path, image_latent=None, iterations=3000, log_freq=25):
    latent_vector = torch.randn(1, 512)
    latent_vector = nn.Parameter(latent_vector)

    optimizer = torch.optim.SGD([latent_vector], lr=1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=10,
        threshold=0.0001,
        threshold_mode="rel",
        cooldown=0,
        min_lr=0,
        eps=1e-08,
        verbose=True,
    )
    loss_fn = torch.nn.MSELoss()

    for iteration in range(iterations):
        optimizer.zero_grad()
        output = model(latent_vector)
        loss = loss_fn(output, image)
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step(loss)
        generated_img = output.clone().detach().cpu()
        # plot_image(generated_img)
        save_image(generated_img, "/home/aryan/sem-5/smai/project/57",
                   "output", iteration + 1)

    return latent_vector.cpu().detach()


def decompressor(model, image_latent, save_path):
    output = model(image_latent)

    generated_img = output.clone().detach().cpu()

    return generated_img.cpu().detach()


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

def plot_image(images):
    """
    Plots the image provided
    :param images: Image as Torch Tensor
    :return: None
    """

    grid = torchvision.utils.make_grid(images.clamp(
        min=-1, max=1), scale_each=True, normalize=True)
    grid_image = grid.permute(1, 2, 0).cpu().numpy()
    plt.imshow(grid_image)
    plt.show()


use_gpu = True if torch.cuda.is_available() else False
print(f"Using GPU: {use_gpu}")
model = torch.hub.load(
    "facebookresearch/pytorch_GAN_zoo:hub", "PGAN", model_name="celebAHQ-512", pretrained=True, useGPU=False
)
generator = model.netG

# Take the image as a float tensor of size 1*3*512*512
image = Image.open("image_1.png")
image = image.convert("RGB")
convert = torchvision.transforms.ToTensor()
image = convert(image)
# PERMUTE
image = image.permute(1, 2, 0)

print("shape", image.shape)
image = image.unsqueeze(0)
print(image.shape)

image = np.transpose(image, (0, 3, 1, 2))
image = image.type(torch.FloatTensor)
print(image.shape)

# image = image.permute((1, 2, 0))
plot_image(image)

save_image(image, "/home/aryan/sem-5/smai/project/57",
           "originalImage", 1)
compressed_image = compressor(
    generator, image, "")
