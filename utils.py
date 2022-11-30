import os
import matplotlib.pyplot as plt


def save_image(image, path, folder, name=""):
    PATH = f"{path}/{folder}"
    if not os.path.exists(PATH):
        os.makedirs(PATH)

    # Remove the first dimension
    image = image.squeeze(0)
    image = image.permute(1, 2, 0).cpu().numpy()
    # Rescale the image to [0, 1]
    image = (image - image.min()) / (image.max() - image.min())
    plt.imshow(image)
#    plt.show()
    # Add iterations
    plt.imsave(f"{PATH}/image_{name}.png", image)
