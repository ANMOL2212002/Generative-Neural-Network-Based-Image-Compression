import tensorflow as tf
import numpy as np
# import utils


def compressor(model, image,  image_latent=None, iterations=2500, log_freq=25):
    # get latent vector using tensorflow
    latent_vector = tf.Variable(tf.random.normal([1, 512]))
    optimizer = tf.keras.optimizers.SGD(learning_rate=1, momentum=0.9)
    loss_fn = tf.keras.losses.MeanSquaredError()
    for iteration in range(iterations):
        with tf.GradientTape() as tape:
            output = model(latent_vector)
            loss = loss_fn(output, image)
        gradients = tape.gradient(loss, latent_vector)
        optimizer.apply_gradients(zip([gradients], [latent_vector]))
        if not iteration % log_freq:
            if isinstance(image_latent, tf.Tensor):
                print(
                    iteration, "Loss:", loss, "MSE (Latent): ", tf.reduce_mean(
                        tf.square(latent_vector - image_latent))
                )
            generated_img = output.clone().detach().cpu()
            # plot_image(generated_img)
            # save_image(generated_img, save_path, "GAN", iteration + 1)
    return latent_vector.cpu().detach()
