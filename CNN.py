import pathlib
import pandas as pd
from sklearn import preprocessing
from tensorboard.plugins.hparams import api as hp
import itertools
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
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


def hp_tuning(hp_list, train_data, train_labels, test_data, test_labels):
    # Create the directory to save log files
    dir = pathlib.Path(__file__).parent.absolute()
    hp_log_dir = dir / 'hparams_tuning'

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
    for hpSet in hp_combinations:
        for (hpValue, hpName) in zip(hpSet, hps_name):
            hp_set_to_test.update({hpName: hpValue})

        print('--- Starting trial: %s' % session_num)
        print({key: value for (key, value) in hp_set_to_test.items()})

        run_name = "/run-%d" % session_num
        accuracy, model = cnn(hp_log_dir + run_name, hp_set_to_test, train_data, train_labels, test_data, test_labels)

        # Return the most accurate model
        if best_model['accuracy'] < accuracy:
            best_model = {'accuracy': accuracy, 'model': model}
        session_num += 1

    return best_model['model']

def cnn(run_dir, hparams, train_data, train_labels, test_data, test_labels):
    with tf.summary.create_file_writer(run_dir).as_default():
        # Record the hyper-parameters used in this trial
        hp.hparams(hparams)

        # Build the ANN
        x = tf.keras.layers.Input(shape=train_data.shape[1:])

        # Layer 1: Just a conventional Conv2D layer
        conv = tf.keras.layers.Conv2D(filters=hparams['conv_filters'], kernel_size=hparams['conv_kernel_size'],
                             strides=hparams['conv_stride'], padding='valid')(x)
        conv = tf.keras.layers.Activation(hparams['conv_activation'])(conv)

        # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
        pcaps = PrimaryCapsule(num_capsule=hparams['pcaps_num_capsule'], dim_capsule=hparams['pcaps_dim_capsule'],
                               kernel_size=hparams['pcaps_kernel_size'], strides=hparams['pcaps_stride'],
                               padding='valid', name='primary_capsule')(conv)

        caps = Capsule(num_capsule=hparams['caps1_num_capsule'], dim_capsule=hparams['caps1_dim_capsule'],
                       routings=hparams['caps1_num_routings'], name='capsule')(pcaps)

        fc1 = tf.keras.layers.Flatten()(caps)
        fc2 = tf.keras.layers.Dense(units=hparams['fc2_num_neurons'])(fc1)
        fc2 = tf.keras.layers.Activation(hparams['fc22_activation'])(fc2)

        # Models for training and evaluation (prediction)
        model = tf.keras.models.Model(inputs=x, outputs=fc2)

        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss=[margin_loss, 'mse'], loss_weights=[0.9995, 0.0005],
                      metrics='accuracy')

        # Training without data augmentation:
        model.fit(x=train_data, y=train_labels, epochs=4, batch_size=100, verbose=1, workers=22,
                  use_multiprocessing=True)

        # Evaluate the ANN
        _, accuracy = model.evaluate(test_data, test_labels)

        # Save accuracy to tensorboard
        tf.summary.scalar('accuracy', accuracy, step=1)

    return accuracy, model

def main():
    (train_data, train_labels), (test_data, test_labels) = load_mnist()

    conv1_filters = hp.HParam('conv_filters', hp.Discrete([4]))
    conv1_kernel_size = hp.HParam('conv_kernel_size', hp.Discrete([9]))
    conv1_stride = hp.HParam('conv_stride', hp.Discrete([1]))
    conv1_activation = hp.HParam('conv_activation', hp.Discrete(['relu']))

    pcaps_num_capsule = hp.HParam('pcaps_num_capsule', hp.Discrete([4]))
    pcaps_dim_capsule = hp.HParam('pcaps_dim_capsule', hp.Discrete([8]))
    pcaps_kernel_size = hp.HParam('pcaps_kernel_size', hp.Discrete([9]))
    pcaps_stride = hp.HParam('pcaps_stride', hp.Discrete([2]))

    caps1_num_capsule = hp.HParam('caps1_num_capsule', hp.Discrete([10]))
    caps1_dim_capsule = hp.HParam('caps1_dim_capsule', hp.Discrete([6]))
    caps1_num_routings = hp.HParam('caps1_num_routings', hp.Discrete([3]))

    fc1_num_neurons = hp.HParam('fc1_num_neurons', hp.Discrete([50]))
    fc1_activation = hp.HParam('fc1_activation', hp.Discrete(['relu']))

    fc2_num_neurons = hp.HParam('fc2_num_neurons', hp.Discrete([10]))
    fc22_activation = hp.HParam('fc22_activation', hp.Discrete(['sigmoid']))

    hp_list = [conv1_filters, conv1_kernel_size, conv1_stride, conv1_activation,
                pcaps_num_capsule, pcaps_dim_capsule, pcaps_kernel_size, pcaps_stride,
                caps1_num_capsule, caps1_dim_capsule, caps1_num_routings, fc2_num_neurons, fc22_activation]

    model = hp_tuning(hp_list, train_data, train_labels, test_data, test_labels)
    model.summary()
    tf.keras.models.save_model(model, 'cnn.h5')

if __name__=='__main__':
    main()