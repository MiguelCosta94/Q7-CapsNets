import tensorflow as tf
import pandas as pd
import pathlib
import time
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from CapsuleLayers import PrimaryCapsule, Capsule, Length, margin_loss


def load_mnist():
    dir = str(pathlib.Path(__file__).parent.absolute()) + "/../Datasets"
    train_db_dir = str(dir) + "/mnist_train.csv"
    test_db_dir = str(dir) + "/mnist_test.csv"
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

    return (train_data, train_labels), (test_data, test_labels)


def latency_per_layer(model, input_data):
    for i, layer in enumerate(model.layers):
        if i == 0:
            input = input_data
        else:
            input = output

        start_time = time.time()
        output = layer(input)
        print(layer.get_config()["name"], ": ", time.time() - start_time)
        output = output.numpy()


def main():
    model = tf.keras.models.load_model("caps_net.h5", custom_objects={'PrimaryCapsule': PrimaryCapsule,
                                                                      'Capsule': Capsule, 'Length': Length,
                                                                      'margin_loss': margin_loss})

    (train_data, train_labels), (test_data, test_labels) = load_mnist()

    data_sample = test_data[0]
    data_sample = data_sample.reshape(-1, 28, 28, 1).astype('float32')
    latency_per_layer(model, data_sample)

if __name__=='__main__':
    main()