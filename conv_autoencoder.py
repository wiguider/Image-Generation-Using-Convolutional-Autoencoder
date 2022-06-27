import os
import pickle

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, ReLU, BatchNormalization, \
    Flatten, Dense, Reshape, Conv2DTranspose, Activation
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np


class Autoencoder:
    """
    Autoencoder represents a Deep Convolutional autoencoder architecture with
    mirrored encoder and decoder components.
    """

    def __init__(self,
                 input_shape,
                 conv_filters,
                 conv_kernels,
                 conv_strides,
                 latent_space_dim):
        self.input_shape = input_shape # [28, 28, 1]
        self.conv_filters = conv_filters # [2, 4, 8]
        self.conv_kernels = conv_kernels # [3, 5, 3]
        self.conv_strides = conv_strides # [1, 2, 2]
        self.latent_space_dim = latent_space_dim # 2

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)
        self._shape_before_bottleneck = None
        self._model_input = None

        self._build()

    def summary(self):
        """Prints the summary of the encoder, decoder, 
        and of the whole autoencoder.
        """
        self.encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        """
        Configures the model for training.

        Args:
            learning_rate (float, optional): The learning rate of the optimizer.
            Defaults to 0.0001.
        """
        # Optimizer that implements the Adam algorithm.
        optimizer = Adam(learning_rate=learning_rate)
        # Loss function
        mse_loss = MeanSquaredError()
        # setting the model's configuration
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self, x_train, batch_size, num_epochs):
        """
        Trains the model for a fixed number of epochs (iterations on a dataset).

        Args:
            x_train (Input data): It could be:
          - A Numpy array (or array-like), or a list of arrays
            (in case the model has multiple inputs).
          - A TensorFlow tensor, or a list of tensors
            (in case the model has multiple inputs).
          - A dict mapping input names to the corresponding array/tensors,
            if the model has named inputs.
          - A `tf.data` dataset. Should return a tuple
            of either `(inputs, targets)` or
            `(inputs, targets, sample_weights)`.
          - A generator or `keras.utils.Sequence` returning `(inputs, targets)`
            or `(inputs, targets, sample_weights)`.
          - A `tf.keras.utils.experimental.DatasetCreator`, which wraps a
            callable that takes a single argument of type
            `tf.distribute.InputContext`, and returns a `tf.data.Dataset`.
            `DatasetCreator` should be used when users prefer to specify the
            per-replica batching and sharding logic for the `Dataset`.
            See `tf.keras.utils.experimental.DatasetCreator` doc for more
            information.
            batch_size (Integer or `None`): Number of samples per gradient update.
            If unspecified, `batch_size` will default to 32.

            num_epochs (Integer): Number of epochs to train the model.
            An epoch is an iteration over the entire `x_train` data provided.
        """
        self.model.fit(x_train,
                       x_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)

    def save(self, folder_path="."):
        """Creates a folder if it does not exist in the given path. 
        And, saves the parameters and weights of the model in `folder_path`.

        Args:
            folder_path (str, optional): Path of the folder.
            Defaults to the path of the current directory.
        """
        self._create_folder_if_it_doesnt_exist(folder_path)
        self._save_parameters(folder_path)
        self._save_weights(folder_path)

    def load_weights(self, weights_path):
        """Loads the weights of the model saved in the given path.

        Args:
            weights_path (str): Path of the file where the weights are saved.
        """
        self.model.load_weights(weights_path)

    def reconstruct(self, images):
        """Given a list of images, extracts their latent representations and reconstructs similar images. 

        Args:
            images (Input images): A list of Numpy 3d arrays.

        Returns:
            reconstructed_images, latent_representations: Respectively, the reconstructed images and their latent representations.
        """
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, folder_path="."):
        """Loads the model from the given folder `folder_path`.

        Args:
            folder_path (str, optional): Path of the folder.
            Defaults to the path of the current directory.

        Returns:
            Autoencoder: The model saved in the given folder.
        """
        parameters_path = os.path.join(folder_path, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        autoencoder = Autoencoder(*parameters)
        weights_path = os.path.join(folder_path, "weights.h5")
        autoencoder.load_weights(weights_path)
        return autoencoder

    def _create_folder_if_it_doesnt_exist(self, folder_path):
        """Creates a folder if it does not exist in the given path.

        Args:
            folder_path (str): Path of the folder.
        """
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

    def _save_parameters(self, folder_path):
        """Saves the parameters of the model.

        Args:
            folder_path (str): Path of the folder where the parameters will be saved.
        """
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_path = os.path.join(folder_path, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, folder_path):
        """Saves the weights of the model.

        Args:
            folder_path (str):  Path of the folder where the weights will be saved.
        """
        file_path = os.path.join(folder_path, "weights.h5")
        self.model.save_weights(file_path)

    def _build(self):
        """Builds the model.
        """
        self._build_encoder()
        self._build_decoder()
        self._build_autoencoder()

    def _build_autoencoder(self):
        """Builds the autoencoder.
        """
        model_input = self._model_input
        model_output = self.decoder(self.encoder(model_input))
        self.model = Model(model_input, model_output, name="autoencoder")

    def _build_decoder(self):
        """Builds the decoder.
        """
        decoder_input = self._add_decoder_input()
        dense_layer = self._add_dense_layer(decoder_input)
        reshape_layer = self._add_reshape_layer(dense_layer)
        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        self.decoder = Model(decoder_input, decoder_output, name="decoder")

    def _add_decoder_input(self):
        """Instantiates a Keras tensor and sets the input shape of the decoder.

        Returns:
            A `tensor`.
        """
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        """Adds a Dense layer to the decoder.

        Args:
            decoder_input (tensor): Input of the decoder

        Returns:
            _type_: _description_
        """
        num_neurons = np.prod(self._shape_before_bottleneck) # [1, 2, 4] -> 8
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        """Reshapes the dense layer containing the latent features.
        """
        return Reshape(self._shape_before_bottleneck)(dense_layer)

    def _add_conv_transpose_layers(self, x):
        """Adds conv transpose blocks."""
        # loop through all the conv layers in reverse order and stop at the
        # first layer
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        """Adds a conv_transpose_layer to the graph of layers in the decoder.

        Args:
            layer_index (Integer): _description_
            x (tensor): The graph of layers in the decoder.

        Returns:
            tensor: The graph of layers in the decoder plus a conv_transpose_layer.
        """
        layer_num = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_num}"
        )
        x = conv_transpose_layer(x)
        x = ReLU(name=f"decoder_relu_{layer_num}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_num}")(x)
        return x

    def _add_decoder_output(self, x):
        """Adds an output layer to the graph of layers in the decoder.

        Args:
            x (tensor): The graph of layers in the decoder.

        Returns:
            tensor: The graph of layers in the decoder plus the output layer.
        """
        conv_transpose_layer = Conv2DTranspose(
            filters=self.input_shape[-1],
            kernel_size=self.conv_kernels[0],
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("sigmoid", name="sigmoid_layer")(x)
        return output_layer

    def _build_encoder(self):
        """Builds the encoder.
        """
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
        bottleneck = self._add_bottleneck(conv_layers)
        self._model_input = encoder_input
        self.encoder = Model(encoder_input, bottleneck, name="encoder")

    def _add_encoder_input(self):
        """Instantiates a Keras tensor and sets the input shape of the encoder.

        Returns:
            A `tensor`.
        """
        return Input(shape=self.input_shape, name="encoder_input")

    def _add_conv_layers(self, encoder_input):
        """Creates all convolutional blocks in encoder."""
        x = encoder_input
        for layer_index in range(self._num_conv_layers):
            x = self._add_conv_layer(layer_index, x)
        return x

    def _add_conv_layer(self, layer_index, x):
        """Adds a convolutional block to a graph of layers, consisting of
        conv 2d + ReLU + batch normalization.
        """
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            filters=self.conv_filters[layer_index],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"encoder_conv_layer_{layer_number}"
        )
        x = conv_layer(x)
        x = ReLU(name=f"encoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        return x

    def _add_bottleneck(self, x):
        """Flatten data and add bottleneck (Dense layer)."""
        self._shape_before_bottleneck = K.int_shape(x)[1:]
        x = Flatten()(x)
        x = Dense(self.latent_space_dim, name="encoder_output")(x)
        return x


if __name__ == "__main__":
    autoencoder = Autoencoder(
        input_shape=(28, 28, 1), # shape of the images in the dataset
        conv_filters=(32, 64, 64, 64),# number of filters in each convolutional layer
        conv_kernels=(3, 3, 3, 3),# number of kernels in each convolutional layer
        conv_strides=(1, 2, 2, 1),# number fo strides in each convolutional layer
        latent_space_dim=2 # dimension of the latent space
    )
    autoencoder.summary()
