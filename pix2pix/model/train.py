import os
import sys
import time
import numpy as np
import models
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import keras.backend as K
# Utils
sys.path.append("../utils")
import general_utils
import data_utils

import glob
import re

from sklearn.model_selection import train_test_split


def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


def resize_2x(x_in):
    n, h, w, d = x_in.shape
    x = np.zeros((n, h*2, w*2, d))
    for i in range(2):
        for j in range(2):
            x[:, i::2, j::2] = x_in
    return x


def resize_half(x_in):
    return x_in[:, 0::2, 0::2]


def load_data():
    full = np.load("../../data/image.npy")
    sketch = np.load("../../data/edge.npy")

    full = full / 255.0 * 2 - 1
    sketch = sketch / 255.0 * 2 - 1

    sketch = sketch

    full_train, full_test = train_test_split(full, test_size=0.1, random_state=42)
    sketch_train, sketch_test = train_test_split(sketch, test_size=0.1, random_state=42)

    return full_train, sketch_train, full_test, sketch_test
    

def train(**kwargs):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    model_name = kwargs["model_name"]
    generator = kwargs["generator"]
    image_data_format = kwargs["image_data_format"]
    img_dim = kwargs["img_dim"]
    patch_size = kwargs["patch_size"]
    bn_mode = kwargs["bn_mode"]
    label_smoothing = kwargs["use_label_smoothing"]
    label_flipping = kwargs["label_flipping"]
    # dset = kwargs["dset"]
    use_mbd = kwargs["use_mbd"]

    continue_training = kwargs["continue_training"]

    epoch_size = n_batch_per_epoch * batch_size

    # Setup environment (logging directory etc)
    general_utils.setup_logging(model_name)

    # Load and rescale data
    # X_full_train, X_sketch_train, X_full_val, X_sketch_val = data_utils.load_data(dset, image_data_format)
    X_full_train, X_sketch_train, X_full_val, X_sketch_val = load_data()
    img_dim = X_full_train.shape[-3:]

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size, image_data_format)

    try:

        # Create optimizers
        opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
        opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Load generator model
        generator_model = models.load("generator_unet_%s" % generator,
                                      img_dim,
                                      nb_patch,
                                      bn_mode,
                                      use_mbd,
                                      batch_size)
        # Load discriminator model
        discriminator_model = models.load("DCGAN_discriminator",
                                          img_dim_disc,
                                          nb_patch,
                                          bn_mode,
                                          use_mbd,
                                          batch_size)

        generator_model.compile(loss='mae', optimizer=opt_discriminator)
        discriminator_model.trainable = False

        DCGAN_model = models.DCGAN(generator_model,
                                   discriminator_model,
                                   img_dim,
                                   patch_size,
                                   image_data_format)

        currentEpoch = 0
        if continue_training:
            print('load_model...')
            getEpoch = lambda file: int(re.sub('.*epoch', '', file).replace('.h5', ''))
            model_dir = "../../models/%s" % model_name

            gen_weights_path = sorted(glob.glob(os.path.join('../../models/%s/gen_weights_epoch*.h5' % (model_name))), key=getEpoch)[-1]
            generator_model.load_weights(gen_weights_path)

            print("gen_weights_path:", gen_weights_path)

            disc_weights_path = sorted(glob.glob(os.path.join('../../models/%s/disc_weights_epoch*.h5' % (model_name))), key=getEpoch)[-1]
            discriminator_model.load_weights(disc_weights_path)
            print("disc_weights_path:", disc_weights_path)

            DCGAN_weights_path = sorted(glob.glob(os.path.join('../../models/%s/DCGAN_weights_epoch*.h5' % (model_name))), key=getEpoch)[-1]
            DCGAN_model.load_weights(DCGAN_weights_path)
            print("DCGAN_weights_path:", DCGAN_weights_path)

            currentEpoch = sorted(map(getEpoch, [gen_weights_path, disc_weights_path, DCGAN_weights_path]))[-1]


        loss = [l1_loss, 'binary_crossentropy']
        loss_weights = [4E1, 1]
        DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

        discriminator_model.trainable = True
        discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

        gen_loss = 100
        disc_loss = 100

        # Start training
        print("Start training")
        for e in range(currentEpoch + 1, nb_epoch):
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            start = time.time()

            for X_full_batch, X_sketch_batch in data_utils.gen_batch(X_full_train, X_sketch_train, batch_size):

                # Create a batch to feed the discriminator model
                X_disc, y_disc = data_utils.get_disc_batch(X_full_batch,
                                                           X_sketch_batch,
                                                           generator_model,
                                                           batch_counter,
                                                           patch_size,
                                                           image_data_format,
                                                           label_smoothing=label_smoothing,
                                                           label_flipping=label_flipping)

                # Update the discriminator
                disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)

                # Create a batch to feed the generator model
                X_gen_target, X_gen = next(data_utils.gen_batch(X_full_train, X_sketch_train, batch_size))
                y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
                y_gen[:, 1] = 1

                # Freeze the discriminator
                discriminator_model.trainable = False
                gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
                # Unfreeze the discriminator
                discriminator_model.trainable = True

                batch_counter += 1
                progbar.add(batch_size, values=[("D logloss", disc_loss),
                                                ("G tot", gen_loss[0]),
                                                ("G L1", gen_loss[1]),
                                                ("G logloss", gen_loss[2])])

                # Save images for visualization
                if batch_counter % 10 == 0:
                    # Get new images from validation
                    data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, generator_model,
                                                    batch_size, image_data_format, "training_%05d" % e)
                    X_full_batch, X_sketch_batch = next(data_utils.gen_batch(X_full_val, X_sketch_val, batch_size, shuffle=False))
                    data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, generator_model,
                                                    batch_size, image_data_format, "validation_%05d" % e)

                if batch_counter >= n_batch_per_epoch:
                    break

            print("")
            print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

            if e % 5 == 0:
                gen_weights_path = os.path.join('../../models/%s/gen_weights_epoch%s.h5' % (model_name, e))
                generator_model.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join('../../models/%s/disc_weights_epoch%s.h5' % (model_name, e))
                discriminator_model.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join('../../models/%s/DCGAN_weights_epoch%s.h5' % (model_name, e))
                DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)

    except KeyboardInterrupt:
        pass
