from tensorflow.keras.datasets import fashion_mnist, mnist

from conv_autoencoder import VAE

LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 100


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def load_fashion_mnist():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def train(x_train, learning_rate, batch_size, epochs):
    # autoencoder = Autoencoder(
    autoencoder = VAE(
            input_shape=(28, 28, 1),  # shape of the images in the dataset
            conv_filters=(32, 64, 64, 64),  # number of filters in each convolutional layer
            conv_kernels=(3, 3, 3, 3),  # number of kernels in each convolutional layer
            conv_strides=(1, 2, 2, 1),  # number fo strides in each convolutional layer
            latent_space_dim=2  # dimension of the latent space
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    x_train, _, _, _ = load_fashion_mnist()
    autoencoder = train(x_train[:100000], LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("variational/model")
