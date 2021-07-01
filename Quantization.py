import pandas as pd
import numpy as np
import tensorflow as tf
import pathlib
import csv
import os
import sys
import time
import re
import Quantization_Backend as qb
from tensorflow.keras.utils import to_categorical
from sklearn import preprocessing
from collections.abc import Iterable
from CapsuleLayers import PrimaryCapsule, Capsule, Length, margin_loss


def load_mnist():
    dir = str(pathlib.Path(__file__).parent.absolute()) + "/../Datasets"
    train_db_dir = str(dir) + "/mnist_train.csv"
    test_db_dir = str(dir) + "/mnist_test.csv"
    quant_db_dir = str(dir) + "/mnist_quantization.csv"
    train_db = pd.read_csv(train_db_dir, delimiter=',')
    test_db = pd.read_csv(test_db_dir, delimiter=',')
    quant_db = pd.read_csv(quant_db_dir, delimiter=',')

    # Get features' values
    train_data = train_db.drop(columns=['label'])
    test_data = test_db.drop(columns=['label'])
    quant_data = quant_db.drop(columns=['label'])
    train_data = train_data.values
    test_data = test_data.values
    quant_data = quant_data.values
    scaler = preprocessing.MaxAbsScaler().fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    quant_data = scaler.transform(quant_data)
    train_data = train_data.reshape(-1, 28, 28, 1).astype('float32')
    test_data = test_data.reshape(-1, 28, 28, 1).astype('float32')
    quant_data = quant_data.reshape(-1, 28, 28, 1).astype('float32')

    # Get labels
    train_labels = train_db['label'].values
    test_labels = test_db['label'].values
    quant_labels = quant_db['label'].values
    train_labels = to_categorical(train_labels.astype('float32'))
    test_labels = to_categorical(test_labels.astype('float32'))
    quant_labels = to_categorical(quant_labels.astype('float32'))

    return (train_data, train_labels), (test_data, test_labels), (quant_data, quant_labels)


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

        else:
            input_range, output_range = qb.get_layer_io_range(layer, input)
            entry = {"layer": layer.get_config()["name"], "input": input_range, "output": output_range}
            range_per_layer.append(entry)

        output = qb.get_layer_output(layer, input)

    return range_per_layer


def get_act_q_format(model, quant_data):
    act_q_format = []
    io_range_per_layer = get_io_range_per_layer(model, quant_data)

    for io_range in io_range_per_layer:
        if ('conv2d' in io_range['layer']) or ('dense' in io_range['layer']):
            entry = qb.get_act_q_format_std_layer(io_range)
            act_q_format.append(entry)

        elif 'primary_capsule' in io_range['layer']:
            entry = qb.get_act_q_format_pcap(io_range)
            act_q_format.append(entry)

        elif 'capsule' in io_range['layer']:
            entry = qb.get_act_q_format_cap(io_range, model.get_layer(io_range["layer"]).routings)
            act_q_format.append(entry)

    return act_q_format


def quantize_wt(model):
    wt_q_list = []
    wt_q_format_list = []

    for layer in model.layers:
        wt = layer.get_weights()
        if len(wt) > 0:
            wt = wt[0]        # Index 0 for weights, index 1 for bias

            if 'dense' in layer.get_config()['name']:
                wt = wt.transpose()
            elif ('conv2d' in layer.get_config()['name']) or ('primary_capsule' in layer.get_config()['name']):
                wt = wt.transpose(3, 0, 1, 2)

            wt = wt.flatten()
            min_val = np.min(wt)
            max_val = np.max(wt)
            qm, qn = qb.get_q_format(min_val, max_val)
            wt_q_format = {"layer": layer.get_config()["name"], "int_bits": qm, "frac_bits": qn}
            wt_q_format_list.append(wt_q_format)

            wt_q = qb.quantize(wt, qn)
            entry = {"layer": layer.get_config()["name"], "wt": wt_q}
            wt_q_list.append(entry)

    return wt_q_list, wt_q_format_list


def quantize_bias(model):
    bias_q_list = []
    bias_q_format_list = []

    for layer in model.layers:
        wt = layer.get_weights()
        if len(wt) > 0:
            if len(wt) > 1:
                bias = wt[1]        # Index 0 for weights, index 1 for bias
                min_val = np.min(bias)
                max_val = np.max(bias)
                qm, qn = qb.get_q_format(min_val, max_val)
                bias_q = qb.quantize(bias, qn)
                bias_q_format = {"layer": layer.get_config()["name"], "int_bits": qm, "frac_bits": qn}
                bias_q_dict = {"layer": layer.get_config()["name"], "bias": bias_q}
                bias_q_format_list.append(bias_q_format)
                bias_q_list.append(bias_q_dict)

            else:
                bias_q_format = {"layer": layer.get_config()["name"], "int_bits": None, "frac_bits": None}
                bias_q_dict = {"layer": layer.get_config()["name"], "bias": None}
                bias_q_format_list.append(bias_q_format)
                bias_q_list.append(bias_q_dict)

    return bias_q_list, bias_q_format_list


def quantize_dataset(data):
    min_val = np.min(data)
    max_val = np.max(data)
    qm, qn = qb.get_q_format(min_val, max_val)
    data = qb.quantize(data, qn)
    data = np.reshape(data, newshape=(np.shape(data)[0], -1))

    return data


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


def get_bias_shift(bias_q_format_list, act_q_format_list, wt_q_format_list, model):
    shift_list = []

    for bias_q_format in bias_q_format_list:
        if bias_q_format["frac_bits"] is not None:
            act_q_format = qb.search_dictionaries("layer", bias_q_format["layer"], act_q_format_list)
            wt_q_format = qb.search_dictionaries("layer", bias_q_format["layer"], wt_q_format_list)

            shift = act_q_format["input"]["frac_bits"] + wt_q_format["frac_bits"] - bias_q_format["frac_bits"]
            entry = {"layer": bias_q_format["layer"], "shift": shift}
            shift_list.append(entry)

    return shift_list


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


def list_to_csv(data, filename):
    dir = str(pathlib.Path(__file__).parent.absolute()) + "/logs"

    if not os.path.exists(dir):
        os.makedirs(dir)

    dir += "/" + filename
    np.set_printoptions(threshold=sys.maxsize)

    with open(dir, 'w', newline='') as fp:
        for sample in data:
            if isinstance(sample, Iterable):
                writer = csv.writer(fp, delimiter=',', lineterminator=',\n', quoting=csv.QUOTE_MINIMAL)
                writer.writerow(sample)
            else:
                fp.write(str(sample) + '\n\n')


def get_layer_size(model):

    print("WEIGHTS")
    for layer in model.layers:
        wt = layer.get_weights()
        if len(wt) > 0:
            wt = wt[0]        # Index 0 for weights, index 1 for bias
            print("Layer: ", layer.get_config()['name'], " Size: ", sys.getsizeof(wt))

    print("BIAS")
    for layer in model.layers:
        wt = layer.get_weights()
        if len(wt) > 0:
            if len(wt) > 1:
                bias = wt[1]        # Index 0 for weights, index 1 for bias
                print("Layer: ", layer.get_config()['name'], " Size: ", sys.getsizeof(bias))



def main():
    start_time = time.time()
    model = tf.keras.models.load_model("caps_net.h5", custom_objects={'PrimaryCapsule': PrimaryCapsule,
                                                                      'Capsule': Capsule, 'Length': Length,
                                                                      'margin_loss': margin_loss})

    (train_data, train_labels), (test_data, test_labels), (quant_data, quant_labels) = load_mnist()
    act_q_format_list = get_act_q_format(model, test_data)
    wt_q_list, wt_q_format_list = quantize_wt(model)
    bias_q_list, bias_q_format_list = quantize_bias(model)
    output_shift_list = get_output_shift(act_q_format_list, wt_q_format_list, model)
    bias_shift_list = get_bias_shift(bias_q_format_list, act_q_format_list, wt_q_format_list, model)
    test_data_q = quantize_dataset(test_data)

    dict_to_header_file(wt_q_list, "wt", "wt_q.h")
    dict_to_csv(wt_q_format_list, "wt_q_format.csv")
    dict_to_header_file(bias_q_list, "bias", "bias_q.h")
    dict_to_csv(bias_q_format_list, "bias_q_format.csv")
    list_to_csv(test_data_q, "test_data_q.csv")
    dict_to_csv(act_q_format_list, "act_q_format.csv")
    dict_to_csv(output_shift_list, "output_shift.csv")
    dict_to_csv(bias_shift_list, "bias_shift.csv")

    print("Total time: ", time.time()-start_time)

    get_layer_size(model)

if __name__=='__main__':
    main()
