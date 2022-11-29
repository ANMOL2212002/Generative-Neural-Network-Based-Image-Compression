import torch
import torch.nn as nn
import pytorch_ssim
from utils import *
from ssim import *


def compressor(generator, save_path, mode,  image, iterations=1000):
    PATH = f"{save_path}/{mode}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_vector = torch.randn(1, 512, device=device)
    latent_vector = nn.Parameter(latent_vector)

    optimizer = torch.optim.SGD([latent_vector], lr=1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
    )
    # Use SSIM as loss function
    loss_fn = SSIM()

    # loss_fn = torch.nn.MSELoss()
    losses = []

    for iteration in range(iterations+1):
        optimizer.zero_grad()
        output = generator(latent_vector)
        loss = 1 - loss_fn(output.cuda(), image.cuda())

        # loss = loss_fn(output, image.cuda())

        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step(loss)
        # Append loss as a float to the list
        losses.append(loss.item())
        generated_image = output.clone().detach().cpu()
        if iteration % 20 == 0:
            print(f"Iteration: {iteration}, Loss: {loss}")
            save_image(generated_image, save_path, mode, 1 + iteration)
    # make a line graph of loss vs iteration
# start a fresh figure
    plt.figure()
    plt.plot(losses)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title(f"Loss vs Iteration for {mode}")
    plt.savefig(f"{save_path}/loss.png")
    return latent_vector.cpu().detach()
