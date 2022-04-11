import tensorflow_model_optimization as tfmot
import tensorflow as tf
import pandas as pd
import numpy as np
import pathlib
import tempfile
from sklearn import preprocessing
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
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


def main():
    (train_data, train_labels), (test_data, test_labels) = load_mnist()

    model = tf.keras.models.load_model("caps_net.h5", custom_objects={'PrimaryCapsule': PrimaryCapsule,
                                                                      'Capsule': Capsule, 'Length': Length,
                                                                      'margin_loss': margin_loss})

    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

    # Compute end step to finish pruning after 2 epochs.
    batch_size = 100
    epochs = 2
    validation_split = 0.1  # 10% of training set will be used for validation set.
    num_images = train_data.shape[0] * (1 - validation_split)
    end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

    log_dir = tempfile.mkdtemp()
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        # Log sparsity and other metrics in Tensorboard.
        tfmot.sparsity.keras.PruningSummaries(log_dir=log_dir)
    ]

    # Define model for pruning.
    pruning_params = {'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
                        initial_sparsity=0.50, final_sparsity=0.80, begin_step=0, end_step=end_step)}

    model_for_pruning = prune_low_magnitude(model, **pruning_params)

    # `prune_low_magnitude` requires a recompile.
    model_for_pruning.compile(optimizer=optimizers.Adam(lr=0.001), loss=[margin_loss, 'mse'],
                              loss_weights=[0.9995, 0.0005], metrics='accuracy')

    model_for_pruning.fit(train_data, train_labels, callbacks=callbacks, epochs=4, batch_size=100, verbose=1,
                          workers=6, use_multiprocessing=True)
    _, accuracy = model_for_pruning.evaluate(test_data, test_labels)

    tf.keras.models.save_model(model_for_pruning, 'caps_net_light.h5')

    model_for_pruning.summary()


if __name__=='__main__':
    main()



