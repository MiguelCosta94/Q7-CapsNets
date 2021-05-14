import numpy as np


def get_q_format(data):
    # Quantize input data of the first layer
    min_val = np.min(data)
    max_val = np.max(data)
    # Find number of integer bits to represent this range
    if (min_val == 0) and (max_val == 0):
        int_bits = 0
    else:
        int_bits = int(np.ceil(np.log2(max(abs(min_val), abs(max_val)))))
    # Remaining bits are fractional bits (1-bit for sign)
    if int_bits < 0:
        int_bits = 0
    frac_bits = 7 - int_bits

    return int_bits, frac_bits


def quantize(data, frac_bits):
    # Floating point data is scaled and rounded to [-128,127]
    data = np.round(data * (2 ** frac_bits))
    data = data.astype(int)
    data = np.minimum(data, 127)
    data = np.maximum(data, -127)

    return data


def get_layer_output(layer, input_data):
    data = np.split(input_data, 1000)
    output = []

    for data_batch in data:
        output_batch = layer(data_batch)
        output_batch = output_batch.numpy()
        if len(output) > 0:
            output = np.append(output, output_batch, axis=0)
        else:
            output = output_batch

    return output


def get_pcap_detailed_output(layer, input_data):
    data = np.split(input_data, 1000)
    output = []
    output_ns = []

    for data_batch in data:
        output_batch = layer(data_batch)
        output_batch = output_batch.numpy()
        output_batch_ns = layer.get_output_ns()
        output_batch_ns = output_batch_ns.numpy()

        if len(output) > 0:
            output = np.append(output, output_batch, axis=0)
            output_ns = np.append(output_ns, output_batch_ns, axis=0)
        else:
            output = output_batch
            output_ns = output_batch_ns

    return output, output_ns


def get_cap_detailed_output(layer, input_data):
    data = np.split(input_data, 1000)

    for i, data_batch in enumerate(data):
        output_batch = layer(data_batch)
        input_hat_batch = layer.get_input_hat()

        output_batch = output_batch.numpy()
        input_hat_batch = input_hat_batch.numpy()

        if i > 0:
            output = np.append(output, output_batch, axis=0)
            input_hat = np.append(input_hat, input_hat_batch, axis=0)
        else:
            output = output_batch
            input_hat = input_hat_batch

    return output, input_hat

def get_act_q_format_std_layer(io_layer):
    qm, qn = get_q_format(io_layer["input"])
    qmn_input = {"int_bits": qm, "frac_bits": qn}
    qm, qn = get_q_format(io_layer["output"])
    qmn_output = {"int_bits": qm, "frac_bits": qn}
    fmt = {"layer": io_layer['layer'], "input": qmn_input, "output": qmn_output}

    return fmt


def get_act_q_format_pcap(io_layer):
    qm, qn = get_q_format(io_layer["input"])
    qmn_input = {"int_bits": qm, "frac_bits": qn}
    qm, qn = get_q_format(io_layer['output'])
    qmn_output = {"int_bits": qm, "frac_bits": qn}
    qm, qn = get_q_format(io_layer['output_ns'])
    qmn_output_ns = {"int_bits": qm, "frac_bits": qn}
    fmt = {"layer": io_layer['layer'], "input": qmn_input, "output_ns": qmn_output_ns, "output": qmn_output}

    return fmt


def get_act_q_format_cap(io_layer):
    qm, qn = get_q_format(io_layer["input"])
    qmn_input = {"int_bits": qm, "frac_bits": qn}
    qm, qn = get_q_format(io_layer['input_hat'])
    qmn_input_hat = {"int_bits": qm, "frac_bits": qn}
    fmt = {"layer": io_layer['layer'], "input": qmn_input, "input_hat": qmn_input_hat}

    qm, qn = get_q_format(io_layer['output'])
    qmn_output = {"int_bits": qm, "frac_bits": qn}
    fmt["output"] = qmn_output

    return fmt


def search_dictionaries(key, value, list_of_dicts):
    for dictio in list_of_dicts:
        if dictio[key] == value:
            return dictio
