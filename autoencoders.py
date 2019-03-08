import numpy as np

from keras import metrics
from keras import callbacks
from keras import backend as K
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Lambda
from keras.layers import Input
from keras.layers import Dense
from keras.models import Model

from sklearn import model_selection


class DAE:
    def __init__(self, input_dim, batch_size, latent_dim):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim
        input_img = Input(shape=(self.input_dim, ))

        # 'encoded' is the encoded representation of the input
        encoded = Dense(
            int(self.input_dim / 2),
            kernel_initializer='glorot_uniform')(input_img)

        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)

        encoded = Dense(
            int(self.input_dim / 4),
            kernel_initializer='glorot_uniform')(encoded)

        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)

        encoded = Dense(
            int(self.input_dim / 8),
            kernel_initializer='glorot_uniform')(encoded)

        encoded = BatchNormalization()(encoded)
        encoded = Activation('relu')(encoded)

        encoded = Dense(self.latent_dim, activation='linear')(encoded)

        # 'decoded' is the lossy reconstruction of the input
        decoded = Dense(
            int(self.input_dim / 8),
            kernel_initializer='glorot_uniform')(encoded)

        decoded = BatchNormalization()(decoded)
        decoded = Activation('relu')(decoded)

        decoded = Dense(
            int(self.input_dim / 4),
            kernel_initializer='glorot_uniform')(decoded)

        decoded = BatchNormalization()(decoded)
        decoded = Activation('relu')(decoded)

        decoded = Dense(
            int(self.input_dim / 2),
            kernel_initializer='glorot_uniform')(decoded)

        decoded = BatchNormalization()(decoded)
        decoded = Activation('relu')(decoded)

        decoded = Dense(
            self.input_dim,
            activation='sigmoid',
            kernel_initializer='glorot_uniform')(decoded)

        self.autoencoder = Model(inputs=input_img, outputs=decoded)
        self.autoencoder.compile(optimizer='Adam', loss='mse')
        self.encoder = Model(inputs=input_img, outputs=encoded)

    # return a fit deep encoder
    def fit(self, x_train, y_train):
        x_train, x_valid = model_selection.train_test_split(
            x_train,
            test_size=max(10, 10 * int(0.01 * x_train.shape[0])),
            stratify=y_train)

        self.autoencoder.fit(
            x_train,
            x_train,
            epochs=9999,
            batch_size=self.batch_size,
            validation_data=(x_valid, x_valid),
            verbose=0,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.0001,
                    patience=10,
                    restore_best_weights=True)
            ])
        return self

    # return prediction for x
    def transform(self, x):
        prediction = self.encoder.predict(x)
        return prediction.reshape((len(prediction),
                                   np.prod(prediction.shape[1:])))


class VAE:
    def __init__(self, input_dim, batch_size, latent_dim):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.latent_dim = latent_dim

        x = Input(batch_shape=(self.batch_size, self.input_dim))
        h = Dense(
            int(self.input_dim / 2), kernel_initializer='glorot_uniform')(x)

        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(
            int(self.input_dim / 4), kernel_initializer='glorot_uniform')(h)

        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(
            int(self.input_dim / 8), kernel_initializer='glorot_uniform')(h)

        h = BatchNormalization()(h)
        h = Activation('relu')(h)

        h = Dense(self.latent_dim, activation='linear')(h)

        z_mean = Dense(self.latent_dim, name='z_mean')(h)
        z_log_var = Dense(self.latent_dim, name='z_var')(h)

        def sampling(args):
            z_mean, z_log_var = args
            # by default, random_normal has mean=0 and std=1.0
            epsilon = K.random_normal(shape=(self.batch_size, self.latent_dim))
            return z_mean + K.exp(z_log_var * 0.5) * epsilon

        # note that 'output_shape' isn't necessary with the TensorFlow backend
        z = Lambda(
            sampling, output_shape=(self.latent_dim, ),
            name='sampling')([z_mean, z_log_var])

        # we instantiate these layers separately so as to reuse them later
        decoder_h = Dense(
            int(self.input_dim / 8), kernel_initializer='glorot_uniform')(z)

        decoder_h = BatchNormalization()(decoder_h)
        decoder_h = Activation('relu')(decoder_h)

        decoder_h = Dense(
            int(self.input_dim / 4),
            kernel_initializer='glorot_uniform')(decoder_h)

        decoder_h = BatchNormalization()(decoder_h)
        decoder_h = Activation('relu')(decoder_h)

        decoder_h = Dense(
            int(self.input_dim / 2),
            kernel_initializer='glorot_uniform')(decoder_h)

        decoder_h = BatchNormalization()(decoder_h)
        decoder_h = Activation('relu')(decoder_h)

        x_decoded_mean = Dense(
            self.input_dim,
            activation='sigmoid',
            kernel_initializer='glorot_uniform')(decoder_h)

        def vae_loss(x, x_decoded_mean):
            reconstruction_loss = metrics.mse(x,
                                              x_decoded_mean) * self.input_dim
            kl_loss = -0.5 * K.mean(
                1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
            return reconstruction_loss + kl_loss

        self.autoencoder = Model(inputs=x, outputs=x_decoded_mean)
        self.autoencoder.compile(optimizer='Adam', loss=vae_loss)
        self.encoder = Model(inputs=x, outputs=z_mean)

    def fit(self, x_train, y_train):
        x_train, x_valid = model_selection.train_test_split(
            x_train,
            test_size=max(10, 10 * int(0.01 * x_train.shape[0])),
            stratify=y_train)

        self.autoencoder.fit(
            x_train,
            x_train,
            epochs=9999,
            batch_size=self.batch_size,
            validation_data=(x_valid, x_valid),
            verbose=0,
            callbacks=[
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    min_delta=0.01,
                    patience=10,
                    restore_best_weights=True)
            ])
        return self

    # return prediction for x
    def transform(self, x):
        prediction = self.encoder.predict(x)
        return prediction.reshape((len(prediction),
                                   np.prod(prediction.shape[1:])))