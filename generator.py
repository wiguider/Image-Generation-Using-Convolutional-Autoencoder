import matplotlib.pyplot as plt
import numpy as np

from conv_autoencoder import VAE
from train_conv_autoencoder import load_fashion_mnist


def select_images(images, labels, num_images=10):
    """selects a random sample from the given images.

    Args:
        images (Input images): List of 3d numpy arrays representing the images.
        labels (Input labels): List of the labels corresponding to the images.
        num_images (int, optional): Number of the images in the sample. Defaults to 10.

    Returns:
        tuple: sample_images, sample_labels
    """
    sample_images_index = np.random.choice(range(len(images)), num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images, sample_labels


def plot_reconstructed_images(images, reconstructed_images):
    """Plots the original images and the ones reconstructed by the autoencoder.

    Args:
        images (Input images): List of 3d numpy arrays representing the original images.
        reconstructed_images (Reconstucted Images): List of 3d numpy arrays representing the images reconstructed by the autoencoder.
    """
    fig = plt.figure(figsize=(15, 3))
    num_images = len(images)
    for i, (image, reconstructed_image) in enumerate(zip(images, reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2, num_images, i + 1)
        ax.axis("off")
        ax.imshow(image, cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2, num_images, i + num_images + 1)
        ax.axis("off")
        ax.imshow(reconstructed_image, cmap="gray_r")
    plt.show()


def plot_images_encoded_in_latent_space(latent_representations, sample_labels):
    """Plots the latent representations of the images.

    Args:
        latent_representations (List): latent representations of the images-
        sample_labels (List): labels of the images.
    """
    plt.figure(figsize=(10, 10))
    plt.scatter(latent_representations[:, 0],
                latent_representations[:, 1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    autoencoder = VAE.load("variational/model")
    x_train, y_train, x_test, y_test = load_fashion_mnist()  # load_mnist()

    num_sample_images_to_show = 8
    for i in np.arange(9):
        idx = np.where(y_test == i)
        sample_images, _ = select_images(x_test[idx], y_test[idx], num_sample_images_to_show)
        reconstructed_images, _ = autoencoder.reconstruct(sample_images)
        plot_reconstructed_images(sample_images, reconstructed_images)

    num_images = 6000
    sample_images, sample_labels = select_images(x_test, y_test, num_images)
    _, latent_representations = autoencoder.reconstruct(sample_images)
    plot_images_encoded_in_latent_space(latent_representations, sample_labels)
