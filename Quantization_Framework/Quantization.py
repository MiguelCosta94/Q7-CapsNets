import pandas as pd
import numpy as np
import tensorflow as tf
import pathlib
import csv
import os
import sys
import time
import re
import pickle
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from CapsuleLayers import PrimaryCapsule, Capsule, Length, margin_loss
from smallnorb_utils import SmallNORBDataset
import Quantization_Backend as qb


def load_mnist():
    dir = os.path.join(os.path.dirname(__file__), '../Datasets/mnist/')
    test_db_dir = str(dir) + "mnist_test.csv"
    test_db = pd.read_csv(test_db_dir, delimiter=',')

    # Get features' values
    test_data = test_db.drop(columns=['label'])
    test_data = test_data.values / 255
    test_data = test_data.reshape(-1, 28, 28, 1).astype('float32')

    # Get labels
    test_labels = test_db['label'].values

    return test_data, test_labels


def batch_dataset_to_numpy(ds):
    dataset = []

    for sample in tqdm(ds, total=len(ds)):
        label = sample.category

        image_lt = sample.image_lt/255
        image_lt = np.expand_dims(image_lt, axis=-1)
        image_lt = tf.image.resize(image_lt, (32, 32))
        image_rt = sample.image_rt/255
        image_rt = np.expand_dims(image_rt, axis=-1)
        image_rt = tf.image.resize(image_rt, (32, 32))

        image_ch = np.stack((image_lt, image_rt), axis=2)
        image_ch = np.reshape(image_ch, (32,32,2))

        dataset.append((image_ch, label))

    data, labels = zip(*dataset)

    return np.array(data), np.array(labels)


def load_small_norb():
    dataset = SmallNORBDataset(dataset_root='../Datasets/smallnorb')
    smallnorb_train = dataset.data['train']
    smallnorb_test = dataset.data['test']

    train_data, train_labels = batch_dataset_to_numpy(smallnorb_train)
    test_data, test_labels = batch_dataset_to_numpy(smallnorb_test)

    train_data_shape = train_data.shape
    test_data_shape = test_data.shape
    train_data = np.reshape(train_data, (train_data_shape[0], -1))
    test_data = np.reshape(test_data, (test_data_shape[0], -1))
    scaler = StandardScaler()
    scaler = scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    train_data = np.reshape(train_data, train_data_shape)
    test_data = np.reshape(test_data, test_data_shape)

    return test_data, test_labels


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data

def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10(negatives=False):
    data_dir = os.path.join(os.path.dirname(__file__), '../Datasets/cifar-10-batches-py')

    cifar_test_data_dict = unpickle(data_dir + "/test_batch")
    cifar_test_data = cifar_test_data_dict[b'data']
    cifar_test_labels = cifar_test_data_dict[b'labels']

    cifar_test_data = cifar_test_data.reshape((len(cifar_test_data), 3, 32, 32))
    if negatives:
        cifar_test_data = cifar_test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        cifar_test_data = np.rollaxis(cifar_test_data, 1, 4)
    cifar_test_labels = np.array(cifar_test_labels)

    cifar_test_data = cifar_test_data / 255

    return cifar_test_data, cifar_test_labels


def get_io_range_per_layer(model, quant_data):
    range_per_layer = []

    for i, layer in enumerate(model.layers):
        if i == 0:
            input = quant_data
        else:
            input = output

        if 'primary_capsule' in layer.get_config()['name']:
            input_range, output_range, output_ns_range = qb.get_pcap_io_range(layer, input)
            entry = {"layer": layer.get_config()["name"], "input": input_range, "output_ns": output_ns_range,
                     "output": output_range}
            range_per_layer.append(entry)

        elif 'capsule' in layer.get_config()['name']:
            input_range, output_range, input_hat_range, output_ns_range, output_s_range, cc_range, b_inst_range,\
                b_new_range, b_old_range = qb.get_cap_io_range(layer, input)
            
            entry = {"layer": layer.get_config()["name"], "input": input_range, "input_hat": input_hat_range,
                     "output_ns": output_ns_range, "output_s": output_s_range, "cc": cc_range,
                     "b_inst": b_inst_range, "b_old": b_old_range, "b_new": b_new_range, "output": output_range}

            range_per_layer.append(entry)

        elif ('conv2d' in layer.get_config()['name']) or ('dense' in layer.get_config()['name']):
            input_range, output_range = qb.get_layer_io_range(layer, input)

            if len(model.layers) > i+1:
                if model.layers[i+1].get_config()['activation'] == 'relu':
                    if output_range['min'] < 0:
                        output_range['min'] = 0
                    if output_range['max'] < 0:
                        output_range['max'] = 0
                        
            entry = {"layer": layer.get_config()["name"], "input": input_range, "output": output_range}
            range_per_layer.append(entry)

        output = layer(input)

    return range_per_layer


def get_act_q_format(model, quant_data, optimization):
    act_q_format = []
    io_range_per_layer = get_io_range_per_layer(model, quant_data)

    for i, io_range in enumerate(io_range_per_layer):
        if ('conv2d' in io_range['layer']) or ('dense' in io_range['layer']):
            entry = qb.get_act_q_format_std_layer(io_range, optimization)
            act_q_format.append(entry)

        elif 'primary_capsule' in io_range['layer']:
            entry = qb.get_act_q_format_pcap(io_range, optimization)
            act_q_format.append(entry)

        elif 'capsule' in io_range['layer']:
            entry = qb.get_act_q_format_cap(io_range, model.get_layer(io_range["layer"]).routings, optimization)
            act_q_format.append(entry)

    return act_q_format


def quantize_wt(model, optimization, target_platform):
    wt_q_list = []
    wt_q_format_list = []

    for layer in model.layers:
        wt = layer.get_weights()
        if len(wt) > 0:
            wt = wt[0]        # Index 0 for weights, index 1 for bias

            if (target_platform == 'arm') or (target_platform == 'riscv'):
                if 'dense' in layer.get_config()['name']:
                    wt = wt.transpose()
                if ('conv2d' in layer.get_config()['name']) or ('primary_capsule' in layer.get_config()['name']):
                    wt = wt.transpose(3, 0, 1, 2)

            wt = wt.flatten()
            min_val = np.min(wt)
            max_val = np.max(wt)
            qm, qn = qb.get_q_format(min_val, max_val, optimization)
            wt_q_format = {"layer": layer.get_config()["name"], "int_bits": qm, "frac_bits": qn}
            wt_q_format_list.append(wt_q_format)
            wt_q = qb.quantize(wt, qn)
            entry = {"layer": layer.get_config()["name"], "wt": wt_q}
            wt_q_list.append(entry)

    return wt_q_list, wt_q_format_list


def get_bias_q_format(model, optimization):
    bias_q_format_list = []

    for layer in model.layers:
        wt = layer.get_weights()
        if len(wt) > 0:
            if len(wt) > 1:
                bias = wt[1]        # Index 0 for weights, index 1 for bias
                min_val = np.min(bias)
                max_val = np.max(bias)
                qm, qn = qb.get_q_format(min_val, max_val, optimization)
                bias_q_format = {"layer": layer.get_config()["name"], "int_bits": qm, "frac_bits": qn}
                bias_q_format_list.append(bias_q_format)

    return bias_q_format_list


def get_bias_shift(act_q_format_list, wt_q_format_list, bias_q_format_list):
    shift_list = []

    for bias_q_format in bias_q_format_list:
        act_q_format = qb.search_dictionaries("layer", bias_q_format["layer"], act_q_format_list)
        wt_q_format = qb.search_dictionaries("layer", bias_q_format["layer"], wt_q_format_list)

        shift = act_q_format["input"]["frac_bits"] + wt_q_format["frac_bits"] - bias_q_format["frac_bits"]
        
        if shift < 0:
            diff = abs(shift)
            bias_q_format["frac_bits"] -= diff
            shift = 0 

        entry = {"layer": bias_q_format["layer"], "shift": shift}
        shift_list.append(entry)

    return shift_list, bias_q_format_list


def quantize_bias(model, bias_q_format_list):
    bias_q_list = []

    for bias_q_format in bias_q_format_list:
        layer = model.get_layer(name=bias_q_format["layer"])
        wt = layer.get_weights()
        bias = wt[1]        # Index 0 for weights, index 1 for bias
        bias_q = qb.quantize(bias, bias_q_format["frac_bits"])
        bias_q_dict = {"layer": layer.get_config()["name"], "bias": bias_q}
        bias_q_list.append(bias_q_dict)

    return bias_q_list


def get_output_shift(act_q_format_list, wt_q_format_list, model):
    shift_list = []

    for wt_q_format in wt_q_format_list:
        act_q_format = qb.search_dictionaries("layer", wt_q_format["layer"], act_q_format_list)

        if 'primary_capsule' in wt_q_format["layer"]:
            shift = act_q_format["input"]["frac_bits"] + wt_q_format["frac_bits"] - act_q_format["output_ns"]["frac_bits"]
            entry = {"layer": wt_q_format["layer"], "shift": shift}
            shift_list.append(entry)

        elif 'capsule' in wt_q_format["layer"]:
            shift = act_q_format["input"]["frac_bits"] + wt_q_format["frac_bits"] - act_q_format["input_hat"]["frac_bits"]
            entry = {"layer": wt_q_format["layer"], "input_hat_shift": shift}

            num_routings = model.get_layer(wt_q_format["layer"]).routings
            for i in range(num_routings):
                shift = act_q_format["input_hat"]["frac_bits"] + act_q_format["cc_"+str(i)]["frac_bits"] -\
                        act_q_format["output_ns_"+str(i)]["frac_bits"]
                entry["output_shift_"+str(i)] = shift
            for i in range(num_routings-1):
                shift = act_q_format["input_hat"]["frac_bits"] + act_q_format["output_s_"+str(i)]["frac_bits"] -\
                        act_q_format["b_inst_"+str(i)]["frac_bits"]
                entry["b_inst_shift_"+str(i)] = shift

                shift = act_q_format["b_inst_"+str(i)]["frac_bits"] + act_q_format["b_old_"+str(i)]["frac_bits"] -\
                        act_q_format["b_new_"+str(i)]["frac_bits"]
                entry["b_new_shift_"+str(i)] = shift

            shift_list.append(entry)

        else:
            shift = act_q_format["input"]["frac_bits"] + wt_q_format["frac_bits"] - act_q_format["output"]["frac_bits"]
            entry = {"layer": wt_q_format["layer"], "shift": shift}
            shift_list.append(entry)

    return shift_list


def quantize_dataset(data, optimization):
    data = data.flatten()
    min_val = np.min(data)
    max_val = np.max(data)
    qm, qn = qb.get_q_format(min_val, max_val, optimization)
    data = qb.quantize(data, qn)
    data = data.astype('int8')

    return data


def dict_to_header_file(list_of_dict, module_name, file_name):
    np.set_printoptions(threshold=sys.maxsize)

    string = (
        '#ifndef ' + module_name + '\n'
        '#define ' + module_name + '\n\n'
    )

    for dict in list_of_dict:
        if dict[module_name] is not None:
            values = np.array2string(dict[module_name], separator=',')
            values = re.sub('[\[\]\n]', '', values)

            string += (
                '#define ' + dict['layer'].upper() + '_' + module_name.upper() + ' {' + values + '}\n\n'
            )

    string += (
        '#endif\n'
    )

    dir = str(pathlib.Path(__file__).parent.absolute()) + "/logs"
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir += "/" + file_name

    with open(dir, 'w') as f:
        f.write(string)


def dict_to_csv(data, filename):
    dir = str(pathlib.Path(__file__).parent.absolute()) + "/logs"

    if not os.path.exists(dir):
        os.makedirs(dir)

    dir += "/" + filename

    with open(dir, 'w') as fp:
        for sample in data:
            writer = csv.DictWriter(fp, fieldnames=sample, lineterminator='\n')
            writer.writeheader()
            writer.writerow(sample)
            fp.writelines('\n')


def dataset_to_bin(data, labels, data_filename, labels_filename):
    for i, row in enumerate(data):
        row_bin = row.tobytes()

        if i == 0:
            data_byte = bytearray(row_bin)
        else:
            data_byte.extend(row_bin)

    labels = np.array(labels, dtype=np.int8)
    labels_byte = bytearray(labels)

    write_to_bin_file(data_byte, data_filename)
    write_to_bin_file(labels_byte, labels_filename)


def write_to_bin_file(data, file_name):
    dir = str(pathlib.Path(__file__).parent.absolute()) + "/logs"
    if not os.path.exists(dir):
        os.makedirs(dir)
    dir += '/' + file_name

    with open(dir, 'wb') as f:
        f.write(data)


def get_buffer_aux_size(model, mcu_arch, num_cores):
    buffer_aux_max_size = 0
    buffer_aux_size = 0

    for i, layer in enumerate(model.layers):
        if ('primary_capsule' in layer.get_config()['name']) or ('conv2d' in layer.get_config()['name']):
            kernel_size = layer.get_config()['kernel_size']
            if isinstance(kernel_size, int):
                kernel_size = (kernel_size, kernel_size)
            ch_im_in = layer.get_weights()[0].shape[2]
            if mcu_arch == 'arm':
                buffer_aux_size = 2 * 2 * ch_im_in * kernel_size[0] * kernel_size[1]
            elif mcu_arch == 'riscv':
                buffer_aux_size = 2 * num_cores * ch_im_in * kernel_size[0] * kernel_size[1]
        elif ('capsule' in layer.get_config()['name']):
            input_dim_capsule = model.layers[i-1].get_config()['dim_capsule']
            input_num_capsule = layer.get_weights()[0].shape[1]
            num_capsule = layer.get_config()['num_capsule']
            dim_capsule = layer.get_config()['dim_capsule']

            calc_inputs_hat_size = input_dim_capsule
            calc_coupling_coefs_size = num_capsule * input_num_capsule
            calc_caps_output_size = input_num_capsule * dim_capsule
            calc_agreement_w_prev_caps = dim_capsule
            sizes = np.array((calc_inputs_hat_size, calc_coupling_coefs_size, calc_caps_output_size, calc_agreement_w_prev_caps))

            buffer_aux_size = np.max(sizes)

        buffer_aux_max_size = buffer_aux_size if buffer_aux_size > buffer_aux_max_size else buffer_aux_max_size

    return buffer_aux_max_size


def main():
    start_time = time.time()
    optimization = True
    target_platform = 'riscv'     #'arm', 'riscv', or 'tf' for simulator
    num_cores = 1
    
    model = tf.keras.models.load_model("caps_net_mnist.h5", custom_objects={'PrimaryCapsule': PrimaryCapsule,
                                                                      'Capsule': Capsule, 'Length': Length,
                                                                      'margin_loss': margin_loss})

    test_data, test_labels = load_mnist()
    act_q_format_list = get_act_q_format(model, test_data, optimization)
    bias_q_format_list = get_bias_q_format(model, optimization)
    wt_q_list, wt_q_format_list = quantize_wt(model, optimization, target_platform)
    bias_shift_list, bias_q_format_list = get_bias_shift(act_q_format_list, wt_q_format_list, bias_q_format_list)
    bias_q_list = quantize_bias(model, bias_q_format_list)
    output_shift_list = get_output_shift(act_q_format_list, wt_q_format_list, model)
    test_data_q = quantize_dataset(test_data, optimization)

    dict_to_header_file(wt_q_list, "wt", "wt_q.h")
    dict_to_csv(wt_q_format_list, "wt_q_format.csv")
    dict_to_header_file(bias_q_list, "bias", "bias_q.h")
    dict_to_csv(bias_q_format_list, "bias_q_format.csv")
    dataset_to_bin(test_data_q, test_labels, 'data.bin', 'labels.bin')
    dict_to_csv(act_q_format_list, "act_q_format.csv")
    dict_to_csv(output_shift_list, "output_shift.csv")
    dict_to_csv(bias_shift_list, "bias_shift.csv")

    if target_platform != 'tf':
        buffer_aux_size = get_buffer_aux_size(model, target_platform, num_cores)
        print("buffer_aux_size: " + str(buffer_aux_size))

    print("Total time: ", time.time()-start_time)


if __name__=='__main__':
    main()
