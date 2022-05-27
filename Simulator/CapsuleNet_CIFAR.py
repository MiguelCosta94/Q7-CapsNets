import numpy
import tensorflow as tf
import numpy as np
import pathlib
import sys
import math
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.utils import to_categorical
from CapsuleLayers import PrimaryCapsule, Capsule, Length, margin_loss


def load_cifar_10_bin():
    test_data_q_dir = str(pathlib.Path(__file__).parent.absolute()) + "/logs_cifar_tf/data.bin"
    test_labels_dir = str(pathlib.Path(__file__).parent.absolute()) + "/logs_cifar_tf/labels.bin"

    file_data = open(test_data_q_dir, "rb")
    data_buffer = file_data.read(10000*32*32*3)
    test_data = np.frombuffer(data_buffer, dtype='int8')
    test_data = test_data.reshape(-1,32,32,3).astype('float32')

    file_labels = open(test_labels_dir, "rb")
    labels_buffer = file_labels.read(10000)
    test_labels = np.frombuffer(labels_buffer, dtype='int8')
    test_labels = to_categorical(test_labels.astype('float32'))

    return test_data, test_labels


def load_model_wt_bias(file_name):
    first_define = False
    wt_list = []
    dir = str(pathlib.Path(__file__).parent.absolute()) + "/logs_cifar_tf/" + file_name

    np.set_printoptions(threshold=sys.maxsize)

    with open(dir, 'r', newline='') as fp:
        for line in fp:
            if ("#define" in line) and (first_define==False):
                first_define=True
            elif ("#define" in line) and (first_define==True):
                start = "{"
                end = "}"
                wt_str = line[line.find(start)+1 : line.find(end)]
                wt = [int(i) for i in wt_str.split(',')]
                wt_list.append(wt)

    return wt_list


def set_wt_bias(model, wt_list, bias_list, bias_shifts, nn_rounds):
    wt_index = 0
    bias_index = 0
    bias_shift_index = 0

    for layer in model.layers:
        wt = layer.get_weights()
        if ('conv2d' in layer.get_config()['name']) or ('capsule' in layer.get_config()['name']) or ('dense' in layer.get_config()['name']):
            if len(wt) > 1:
                wt_q = np.array(wt_list[wt_index])
                wt_q = np.reshape(wt_q, wt[0].shape)
                wt_q = wt_q.astype('float32')
                wt_index += 1

                bias_q = np.array(bias_list[bias_index])
                bias_q = np.reshape(bias_q, wt[1].shape)
                bias_q = bias_q.astype('float32')
                bias_q = np.multiply(bias_q, math.pow(2, bias_shifts[bias_shift_index]))
                bias_q += math.pow(2.0, nn_rounds[bias_shift_index]-1.0)
                bias_index += 1
                bias_shift_index += 1

                params = numpy.array([wt_q, bias_q])
                layer.set_weights(params)

            elif len(wt) > 0:
                wt_q = np.array(wt_list[wt_index])
                wt_q = np.reshape(wt_q, wt[0].shape)
                wt_q = wt_q.astype('float32')
                layer.set_weights(wt_q)
                wt_index += 1

    return model


def get_params():
    conv1_params = {'filters': 32, 'kernel_size': 3, 'stride': 1, 'activation': 'relu', 'out_shift': 10}
    conv2_params = {'filters': 32, 'kernel_size': 3, 'stride': 1, 'activation': 'relu', 'out_shift': 9}
    conv3_params = {'filters': 64, 'kernel_size': 3, 'stride': 2, 'activation': 'relu', 'out_shift': 8}
    conv4_params = {'filters': 64, 'kernel_size': 3, 'stride': 2, 'activation': 'relu', 'out_shift': 9}

    pcaps_params = {'num_capsule': 16, 'dim_capsule': 4, 'kernel_size': 3, 'stride': 2,
                                                'squash_in_qn': 1, 'squash_out_qn': 7, 'out_shift': 10}

    caps_params = {'num_capsule': 10, 'dim_capsule': 5, 'num_routings': 3, 
                    'in_hat_shift': 7, 'out_ns_shifts': [6,7,8], 'b_inst_shifts': [8,8], 'b_new_shifts': [7,6],
                                                            'squash_in_qn': [7,6,5], 'squash_out_qn': [8,7,7]}

    params = {'conv1': conv1_params, 'conv2': conv2_params, 'conv3': conv3_params, 'conv4': conv4_params,
                                                                 'pcaps': pcaps_params, 'caps': caps_params}

    bias_shifts = [6, 4, 3, 3, 3]
    nn_rounds = [conv1_params['out_shift'], conv2_params['out_shift'], conv3_params['out_shift'],
                                                    conv4_params['out_shift'], pcaps_params['out_shift']]

    return params, bias_shifts, nn_rounds


def caps_net(params):
    # Build the ANN
    x = tf.keras.layers.Input(shape=(32,32,3))

    # Layer 1: Just a conventional Conv2D layer
    conv1 = layers.Conv2D(filters=params['conv1']['filters'], kernel_size=params['conv1']['kernel_size'],
                         strides=params['conv1']['stride'], padding='valid')(x)
    conv1 = tf.cast(conv1, tf.int32)
    conv1 = tf.bitwise.right_shift(conv1, params['conv1']['out_shift'])
    conv1 = tf.clip_by_value(conv1, -128, 127)
    conv1 = tf.cast(conv1, tf.float32)
    conv1 = layers.Activation(params['conv1']['activation'])(conv1)

    # Layer 2: Just a conventional Conv2D layer
    conv2 = layers.Conv2D(filters=params['conv2']['filters'], kernel_size=params['conv2']['kernel_size'],
                            strides=params['conv2']['stride'], padding='valid')(conv1)
    conv2 = tf.cast(conv2, tf.int32)
    conv2 = tf.bitwise.right_shift(conv2, params['conv2']['out_shift'])
    conv2 = tf.clip_by_value(conv2, -128, 127)
    conv2 = tf.cast(conv2, tf.float32)
    conv2 = layers.Activation(params['conv2']['activation'])(conv2)

    # Layer 3: Just a conventional Conv2D layer
    conv3 = layers.Conv2D(filters=params['conv3']['filters'], kernel_size=params['conv3']['kernel_size'],
                           strides=params['conv3']['stride'], padding='valid')(conv2)
    conv3 = tf.cast(conv3, tf.int32)
    conv3 = tf.bitwise.right_shift(conv3, params['conv3']['out_shift'])
    conv3 = tf.clip_by_value(conv3, -128, 127)
    conv3 = tf.cast(conv3, tf.float32)
    conv3 = layers.Activation(params['conv3']['activation'])(conv3)

    # Layer 4: Just a conventional Conv2D layer
    conv4 = layers.Conv2D(filters=params['conv4']['filters'], kernel_size=params['conv4']['kernel_size'],
                          strides=params['conv4']['stride'], padding='valid')(conv3)
    conv4 = tf.cast(conv4, tf.int32)
    conv4 = tf.bitwise.right_shift(conv4, params['conv4']['out_shift'])
    conv4 = tf.clip_by_value(conv4, -128, 127)
    conv4 = tf.cast(conv4, tf.float32)
    conv4 = layers.Activation(params['conv4']['activation'])(conv4)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    pcaps = PrimaryCapsule(num_capsule=params['pcaps']['num_capsule'], dim_capsule=params['pcaps']['dim_capsule'],
                           kernel_size=params['pcaps']['kernel_size'], strides=params['pcaps']['stride'], padding='valid',
                           squash_in_qn=params['pcaps']['squash_in_qn'], squash_out_qn=params['pcaps']['squash_out_qn'],
                            out_shift=params['pcaps']['out_shift'], name='primary_capsule')(conv4)

    # Layer 3: Capsule layer. Routing algorithm works here
    caps = Capsule(num_capsule=params['caps']['num_capsule'], dim_capsule=params['caps']['dim_capsule'], routings=params['caps']['num_routings'],
                    in_hat_shift=params['caps']['in_hat_shift'], out_ns_shifts=params['caps']['out_ns_shifts'],
                    b_inst_shifts=params['caps']['b_inst_shifts'], b_new_shifts=params['caps']['b_new_shifts'],
                    squash_in_qn=params['caps']['squash_in_qn'], squash_out_qn=params['caps']['squash_out_qn'], name='capsule')(pcaps)

    # Layer 4: Calculates the length of vectors returned by each capsule
    cap_net_out = Length()(caps)
    cap_net_out = tf.cast(cap_net_out, tf.int32)
    cap_net_out = tf.clip_by_value(cap_net_out, -128, 127)
    cap_net_out = tf.cast(cap_net_out, tf.float32)

    # Models for training and evaluation (prediction)
    model = models.Model(inputs=x, outputs=cap_net_out)

    model.compile(optimizer=optimizers.Adam(lr=0.00025), loss=[margin_loss, 'mse'], loss_weights=[0.9995, 0.0005],
                  metrics='accuracy')

    return model


def main():
    test_data, test_labels = load_cifar_10_bin()

    params, bias_shifts, nn_rounds = get_params()
    model = caps_net(params)
    wt_list = load_model_wt_bias("wt_q.h")
    bias_list = load_model_wt_bias("bias_q.h")
    model = set_wt_bias(model, wt_list, bias_list, bias_shifts, nn_rounds)
    _, accuracy = model.evaluate(test_data, test_labels)

    print(accuracy)


if __name__=='__main__':
    main()