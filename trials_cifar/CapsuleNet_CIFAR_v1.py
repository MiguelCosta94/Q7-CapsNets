import tensorflow as tf
import pandas as pd
import pathlib
import itertools
import time
import os
import pickle
import numpy as np
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from tensorboard.plugins.hparams import api as hp
from sklearn import preprocessing
from CapsuleLayers import PrimaryCapsule, Capsule, Length, margin_loss


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10(negatives=False):
    data_dir = os.path.join(os.path.dirname(__file__), '../Datasets/cifar-10-batches-py')

    # training data
    cifar_train_data = None
    cifar_train_labels = []

    for i in range(1, 6):
        cifar_train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            cifar_train_data = cifar_train_data_dict[b'data']
        else:
            cifar_train_data = np.vstack((cifar_train_data, cifar_train_data_dict[b'data']))
        cifar_train_labels += cifar_train_data_dict[b'labels']

    cifar_train_data = cifar_train_data.reshape((len(cifar_train_data), 3, 32, 32))
    if negatives:
        cifar_train_data = cifar_train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_train_data = np.rollaxis(cifar_train_data, 1, 4)
    cifar_train_labels = np.array(cifar_train_labels)

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_labels = np.array(cifar_test_labels)

    cifar_train_data = cifar_train_data / 255
    cifar_test_data = cifar_test_data / 255

    return cifar_train_data, to_categorical(cifar_train_labels), \
        cifar_test_data, to_categorical(cifar_test_labels)


def hp_tuning(hp_list, train_data, train_labels, test_data, test_labels):
    # Create the directory to save log files
    hp_log_dir = pathlib.Path(__file__).parent.absolute() / 'hparams_tuning'
    if not hp_log_dir.exists():
        hp_log_dir.mkdir()
    hp_log_dir = str(hp_log_dir)

    session_num = 0
    best_model = {'accuracy': -1, 'model': -1}
    hps_name = []
    hps_value = []
    hp_set_to_test = {}

    # Get all possible combinations of values for the hyper-parameters
    for index in hp_list:
        hps_name.append(index.name)
    for index in hp_list:
        hps_value.append(index.domain.values)

    hp_combinations = list(itertools.product(*hps_value))
    print("\n--- Total number of combinations for the hyperparameters: ", len(hp_combinations))

    # Hyper-parameter tuning / ANN training
    for hp_set in hp_combinations:
        for (hp_value, hp_name) in zip(hp_set, hps_name):
            hp_set_to_test.update({hp_name: hp_value})

        print('--- Starting trial: %s' % session_num)
        print({key: value for (key, value) in hp_set_to_test.items()})

        run_name = "/run-%d" % session_num
        accuracy, model = caps_net(hp_log_dir + run_name, hp_set_to_test, train_data, train_labels, test_data,
                                   test_labels)
        # Return the most accurate model
        if best_model['accuracy'] < accuracy:
            best_model = {'accuracy': accuracy, 'model': model}
        session_num += 1

    return best_model['model']


def caps_net(run_dir, hparams, train_data, train_labels, test_data, test_labels):
    with tf.summary.create_file_writer(run_dir).as_default():
        # Record the hyper-parameters used in this trial
        hp.hparams(hparams)

        # Build the ANN
        x = tf.keras.layers.Input(shape=train_data.shape[1:])

        # Layer 1: Just a conventional Conv2D layer
        conv1 = layers.Conv2D(filters=hparams['conv1_filters'], kernel_size=hparams['conv1_kernel_size'],
                             strides=hparams['conv1_stride'], padding='same', kernel_regularizer=l2(0.0001))(x)
        conv1 = layers.Activation(hparams['conv1_activation'])(conv1)
        conv1 = layers.BatchNormalization()(conv1)
        conv1 = layers.Dropout(0.1)(conv1)

        conv2 = layers.Conv2D(filters=hparams['conv2_filters'], kernel_size=hparams['conv2_kernel_size'],
                             strides=hparams['conv2_stride'], padding='same', kernel_regularizer=l2(0.0001))(conv1)
        conv2 = layers.Activation(hparams['conv2_activation'])(conv2)
        conv2 = layers.BatchNormalization()(conv2)
        conv2 = layers.Dropout(0.1)(conv2)

        conv3 = layers.Conv2D(filters=hparams['conv3_filters'], kernel_size=hparams['conv3_kernel_size'],
                             strides=hparams['conv3_stride'], padding='same', kernel_regularizer=l2(0.0001))(conv2)
        conv3 = layers.Activation(hparams['conv2_activation'])(conv3)
        conv3 = layers.BatchNormalization()(conv3)
        conv3 = layers.Dropout(0.2)(conv3)

        conv4 = layers.Conv2D(filters=hparams['conv3_filters'], kernel_size=hparams['conv3_kernel_size'],
                             strides=hparams['conv3_stride'], padding='same', kernel_regularizer=l2(0.0001))(conv3)
        conv4 = layers.Activation(hparams['conv2_activation'])(conv4)
        conv4 = layers.BatchNormalization()(conv4)
        conv4 = layers.Dropout(0.2)(conv4)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        pcaps = PrimaryCapsule(num_capsule=hparams['pcaps_num_capsule'], dim_capsule=hparams['pcaps_dim_capsule'],
                               kernel_size=hparams['pcaps_kernel_size'], strides=hparams['pcaps_stride'],
                               padding='same', name='primary_capsule')(conv4)
        pcaps = layers.Dropout(0.3)(pcaps)

        # Layer 3: Capsule layer. Routing algorithm works here
        caps = Capsule(num_capsule=hparams['caps1_num_capsule'], dim_capsule=hparams['caps1_dim_capsule'],
                       routings=hparams['caps1_num_routings'], name='capsule')(pcaps)

        # Layer 4: Calculates the length of vectors returned by each capsule
        cap_net_out = Length()(caps)

        # Models for training and evaluation (prediction)
        model = models.Model(inputs=x, outputs=cap_net_out)

        model.compile(optimizer=optimizers.Adam(lr=0.001), loss=[margin_loss, 'mse'], loss_weights=[0.9995, 0.0005],
                      metrics='accuracy')

        model.summary()

        # Training without data augmentation:
        model.fit(x=train_data, y=train_labels, epochs=100, batch_size=32, verbose=1, workers=6,
                   validation_split=0.1, use_multiprocessing=True)

        # Evaluate the ANN
        _, accuracy = model.evaluate(test_data, test_labels)

        # Save accuracy to tensorboard
        tf.summary.scalar('accuracy', accuracy, step=1)

    return accuracy, model


def main():
    start_time = time.time()
    train_data, train_labels, test_data, test_labels = load_cifar_10()

    # Original CapNet has 256 filters
    conv1_filters = hp.HParam('conv1_filters', hp.Discrete([32]))
    conv1_kernel_size = hp.HParam('conv1_kernel_size', hp.Discrete([3]))
    conv1_stride = hp.HParam('conv1_stride', hp.Discrete([1]))
    conv1_activation = hp.HParam('conv1_activation', hp.Discrete(['relu']))

    conv2_filters = hp.HParam('conv2_filters', hp.Discrete([32]))
    conv2_kernel_size = hp.HParam('conv2_kernel_size', hp.Discrete([3]))
    conv2_stride = hp.HParam('conv2_stride', hp.Discrete([2]))
    conv2_activation = hp.HParam('conv2_activation', hp.Discrete(['relu']))

    conv3_filters = hp.HParam('conv3_filters', hp.Discrete([64]))
    conv3_kernel_size = hp.HParam('conv3_kernel_size', hp.Discrete([3]))
    conv3_stride = hp.HParam('conv3_stride', hp.Discrete([2]))
    conv3_activation = hp.HParam('conv3_activation', hp.Discrete(['relu']))

    conv4_filters = hp.HParam('conv4_filters', hp.Discrete([64]))
    conv4_kernel_size = hp.HParam('conv4_kernel_size', hp.Discrete([3]))
    conv4_stride = hp.HParam('conv4_stride', hp.Discrete([2]))
    conv4_activation = hp.HParam('conv4_activation', hp.Discrete(['relu']))

    # Original CapsNet has 32 pcaps
    pcaps_num_capsule = hp.HParam('pcaps_num_capsule', hp.Discrete([16]))
    pcaps_dim_capsule = hp.HParam('pcaps_dim_capsule', hp.Discrete([4]))
    pcaps_kernel_size = hp.HParam('pcaps_kernel_size', hp.Discrete([3]))
    pcaps_stride = hp.HParam('pcaps_stride', hp.Discrete([2]))

    caps1_num_capsule = hp.HParam('caps1_num_capsule', hp.Discrete([10]))
    caps1_dim_capsule = hp.HParam('caps1_dim_capsule', hp.Discrete([5]))
    caps1_num_routings = hp.HParam('caps1_num_routings', hp.Discrete([3]))

    hp_list = [conv1_filters, conv1_kernel_size, conv1_stride, conv1_activation,
                conv2_filters, conv2_kernel_size, conv2_stride, conv2_activation,
                conv3_filters, conv3_kernel_size, conv3_stride, conv3_activation,
                conv4_filters, conv4_kernel_size, conv4_stride, conv4_activation,
               pcaps_num_capsule, pcaps_dim_capsule, pcaps_kernel_size, pcaps_stride,
               caps1_num_capsule, caps1_dim_capsule, caps1_num_routings]

    model = hp_tuning(hp_list, train_data, train_labels, test_data, test_labels)

    tf.keras.models.save_model(model, 'caps_net_cifar_v1.h5')
    print("Total time: ", time.time() - start_time)


if __name__=='__main__':
    main()