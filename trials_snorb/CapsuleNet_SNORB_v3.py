from posixpath import split
from matplotlib import image
from matplotlib.pyplot import axis
import tensorflow as tf
import pathlib
import itertools
import time
import tensorflow_datasets as tfds
import numpy as np
import random
from tqdm import tqdm
from tensorflow.keras.regularizers import l2
from tensorflow.keras import layers, models, optimizers
from tensorboard.plugins.hparams import api as hp
from CapsuleLayers import PrimaryCapsule, Capsule, Length, margin_loss
from smallnorb_utils import SmallNORBDataset
from sklearn.preprocessing import MaxAbsScaler


batch_size = 32

def batch_dataset_to_numpy(ds):
    dataset = []

    for sample in tqdm(ds, total=len(ds)):
        label = tf.one_hot(sample.category, 5)

        image_lt = sample.image_lt/255
        image_lt = np.expand_dims(image_lt, axis=-1)
        image_lt = tf.image.resize(image_lt, (32, 32))
        image_rt = sample.image_rt/255
        image_rt = np.expand_dims(image_rt, axis=-1)
        image_rt = tf.image.resize(image_rt, (32, 32))

        image_ch = np.stack((image_lt, image_rt), axis=2)
        image_ch = np.reshape(image_ch, (32,32,2))

        dataset.append((image_ch, label))

    data = random.shuffle(dataset)
    data, labels = zip(*dataset)

    return np.array(data), np.array(labels)


def load_small_norb():
    dataset = SmallNORBDataset(dataset_root='../Datasets/smallnorb')
    smallnorb_train = dataset.data['train']
    smallnorb_test = dataset.data['test']

    train_data, train_labels = batch_dataset_to_numpy(smallnorb_train)
    test_data, test_labels = batch_dataset_to_numpy(smallnorb_test)

    #train_data = np.reshape(train_data, (train_data.shape[0], -1))
    #test_data = np.reshape(test_data, (test_data.shape[0], -1))
    #scaler = MaxAbsScaler().fit(train_data)
    #train_data = scaler.transform(train_data)
    #test_data = scaler.transform(test_data)
    #train_data = np.reshape(train_data, (train_data.shape[0], 32, 32, 2))
    #test_data = np.reshape(test_data, (test_data.shape[0], 32, 32, 2))

    return train_data, train_labels, test_data, test_labels


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
        x = tf.keras.layers.Input(shape=(32,32,2))

        # Layer 1: Just a conventional Conv2D layer
        conv = layers.Conv2D(filters=hparams['conv_filters'], kernel_size=hparams['conv_kernel_size'],
                             strides=hparams['conv_stride'], padding='valid', kernel_regularizer=l2(0.0001))(x)
        conv = layers.Activation(hparams['conv_activation'])(conv)
        conv = layers.BatchNormalization()(conv)
        conv = layers.Dropout(0.1)(conv)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        pcaps = PrimaryCapsule(num_capsule=hparams['pcaps_num_capsule'], dim_capsule=hparams['pcaps_dim_capsule'],
                               kernel_size=hparams['pcaps_kernel_size'], strides=hparams['pcaps_stride'],
                               padding='valid', name='primary_capsule')(conv)
        pcaps = layers.Dropout(0.1)(pcaps)

        # Layer 3: Capsule layer. Routing algorithm works here
        caps = Capsule(num_capsule=hparams['caps1_num_capsule'], dim_capsule=hparams['caps1_dim_capsule'],
                       routings=hparams['caps1_num_routings'], name='capsule')(pcaps)

        # Layer 4: Calculates the length of vectors returned by each capsule
        cap_net_out = Length()(caps)

        # Models for training and evaluation (prediction)
        model = models.Model(inputs=x, outputs=cap_net_out)

        model.compile(optimizer=optimizers.Adam(lr=0.00025), loss=[margin_loss, 'mse'], loss_weights=[0.9995, 0.0005],
                      metrics='accuracy')

        model.summary()

        # Training without data augmentation:
        model.fit(x=train_data, y=train_labels, epochs=10, batch_size=32, verbose=1, workers=6,
                   validation_split=0.1, use_multiprocessing=True)

        # Evaluate the ANN
        _, accuracy = model.evaluate(test_data, test_labels)

        # Save accuracy to tensorboard
        tf.summary.scalar('accuracy', accuracy, step=1)

    return accuracy, model


def main():
    start_time = time.time()
    train_data, train_labels, test_data, test_labels = load_small_norb()
    #train_data, train_labels, test_data, test_labels = [], [], [], []

    # Original CapNet has 256 filters
    conv1_filters = hp.HParam('conv_filters', hp.Discrete([16]))
    conv1_kernel_size = hp.HParam('conv_kernel_size', hp.Discrete([5]))
    conv1_stride = hp.HParam('conv_stride', hp.Discrete([1]))
    conv1_activation = hp.HParam('conv_activation', hp.Discrete(['relu']))

    # Original CapsNet has 32 pcaps
    pcaps_num_capsule = hp.HParam('pcaps_num_capsule', hp.Discrete([16]))
    pcaps_dim_capsule = hp.HParam('pcaps_dim_capsule', hp.Discrete([4]))
    pcaps_kernel_size = hp.HParam('pcaps_kernel_size', hp.Discrete([5]))
    pcaps_stride = hp.HParam('pcaps_stride', hp.Discrete([2]))

    caps1_num_capsule = hp.HParam('caps1_num_capsule', hp.Discrete([5]))
    caps1_dim_capsule = hp.HParam('caps1_dim_capsule', hp.Discrete([6]))
    caps1_num_routings = hp.HParam('caps1_num_routings', hp.Discrete([3]))

    hp_list = [conv1_filters, conv1_kernel_size, conv1_stride, conv1_activation,
               pcaps_num_capsule, pcaps_dim_capsule, pcaps_kernel_size, pcaps_stride,
               caps1_num_capsule, caps1_dim_capsule, caps1_num_routings]

    model = hp_tuning(hp_list, train_data, train_labels, test_data, test_labels)

    tf.keras.models.save_model(model, 'caps_net_snorb_v3.h5')
    print("Total time: ", time.time() - start_time)


if __name__=='__main__':
    main()