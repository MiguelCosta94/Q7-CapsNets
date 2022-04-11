import tensorflow as tf
import pandas as pd
import pathlib
import time
import numpy as np
import random
import os
import pickle
from tqdm import tqdm
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from CapsuleLayers import PrimaryCapsule, Capsule, Length, margin_loss
from smallnorb_utils import SmallNORBDataset


def load_mnist():
    dir = os.path.join(os.path.dirname(__file__), '../Datasets/mnist/')
    train_db_dir = str(dir) + "mnist_train.csv"
    test_db_dir = str(dir) + "mnist_test.csv"
    train_db = pd.read_csv(train_db_dir, delimiter=',')
    test_db = pd.read_csv(test_db_dir, delimiter=',')

    # Get features' values
    train_data = train_db.drop(columns=['label'])
    test_data = test_db.drop(columns=['label'])
    train_data = train_data.values
    test_data = test_data.values
    scaler = preprocessing.MaxAbsScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    train_data = train_data.reshape(-1, 28, 28, 1).astype('float32')
    test_data = test_data.reshape(-1, 28, 28, 1).astype('float32')

    # Get labels
    train_labels = train_db['label'].values
    test_labels = test_db['label'].values
    train_labels = to_categorical(train_labels.astype('float32'))
    test_labels = to_categorical(test_labels.astype('float32'))

    return train_data, train_labels, test_data, test_labels


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

    return train_data, train_labels, test_data, test_labels


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



def main():
    model = tf.keras.models.load_model("caps_net_mnist_v2.h5", custom_objects={'PrimaryCapsule': PrimaryCapsule,
                                                                      'Capsule': Capsule, 'Length': Length,
                                                                      'margin_loss': margin_loss})

    train_data, train_labels, test_data, test_labels = load_mnist()

    start_time = time.time()
    _, accuracy = model.evaluate(test_data, test_labels)
    print("Total time: ", time.time() - start_time)

    print("Accuracy: ", accuracy)

if __name__=='__main__':
    main()