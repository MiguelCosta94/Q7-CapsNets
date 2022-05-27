import numpy as np


def get_q_format(min_val, max_val, optimization):
    # Find number of integer bits to represent this range
    if (min_val == 0) and (max_val == 0):
        int_bits = 0
    else:
        int_bits = int(np.ceil(np.log2(max(abs(min_val), abs(max_val)))))
    # Remaining bits are fractional bits (1-bit for sign)
    if int_bits < 0:
        int_bits = 0
    #elif int_bits > 7:
    #    int_bits = 7

    frac_bits = 7 - int_bits

    opt = False
    if optimization:
        max_val_abs = abs(max_val) if abs(max_val) > abs(min_val) else abs(min_val)
        if max_val_abs!=0:
            max_val_q7 = np.round(max_val_abs * (2 ** frac_bits)) 

            while max_val_q7 <= 128:
                opt = True
                frac_bits += 1
                max_val_q7 = np.round(max_val_abs * (2 ** frac_bits))
            
            if opt==True:
                frac_bits = frac_bits - 1

    return int_bits, frac_bits


def quantize(data, frac_bits):
    # Floating point data is scaled and rounded to [-128,127]
    data = np.round(data * (2 ** frac_bits))
    data = data.astype(np.int32)
    data = np.minimum(data, 127)
    data = np.maximum(data, -128)

    return data


def get_layer_output(layer, input_data):
    data = np.split(input_data, 100)
    output = []

    for data_batch in data:
        output_batch = layer(data_batch)
        output_batch = output_batch.numpy()
        if len(output) > 0:
            output = np.append(output, output_batch, axis=0)
        else:
            output = output_batch

    return output


def get_layer_io_range(layer, input_data):
    output_batch = layer(input_data)

    input_data = input_data.numpy()
    input_data = input_data.flatten()
    output_batch = output_batch.numpy()
    output_batch = output_batch.flatten()

    input_range = get_range_io(input_data)
    output_range = get_range_io(output_batch)

    return input_range, output_range


def get_pcap_io_range(layer, input_data):
    output_batch = layer(input_data)
    output_batch_ns = layer.get_output_ns()

    input_data = input_data.numpy()
    input_data = input_data.flatten()
    output_batch = output_batch.numpy()
    output_batch = output_batch.flatten()
    output_batch_ns = output_batch_ns.numpy()
    output_batch_ns = output_batch_ns.flatten()

    input_range = get_range_io(input_data)
    output_range = get_range_io(output_batch)
    output_ns_range = get_range_io(output_batch_ns)

    return input_range, output_range, output_ns_range


def get_cap_io_range(layer, input_data):
    num_rout = layer.routings

    output_batch = layer(input_data)
    input_hat_batch = layer.get_input_hat()
    output_ns_list_batch = layer.get_output_ns_list()
    output_s_list_batch = layer.get_output_s_list()
    cc_list_batch = layer.get_cc_list()
    b_inst_list_batch = layer.get_b_inst_list()
    b_new_list_batch = layer.get_b_new_list()
    b_old_list_batch = layer.get_b_old_list()

    input_data = input_data.numpy()
    input_data = input_data.flatten()
    output_batch = output_batch.numpy()
    output_batch = output_batch.flatten()
    input_hat_batch = input_hat_batch.numpy()
    input_hat_batch = input_hat_batch.flatten()
    output_ns_list_batch = [x.numpy().flatten() for x in output_ns_list_batch]
    output_s_list_batch = [x.numpy().flatten() for x in output_s_list_batch]
    cc_list_batch = [x.numpy().flatten() for x in cc_list_batch]

    if num_rout > 1:
        b_inst_list_batch = [x.numpy().flatten() for x in b_inst_list_batch]
        b_new_list_batch = [x.numpy().flatten() for x in b_new_list_batch]
        b_old_list_batch = [x.numpy().flatten() for x in b_old_list_batch]

    input_range = get_range_io(input_data)
    output_range = get_range_io(output_batch)
    input_hat_range = get_range_io(input_hat_batch)
    output_ns_range = get_range_io_routing(output_ns_list_batch)
    output_s_range = get_range_io_routing(output_s_list_batch)
    cc_range = get_range_io_routing(cc_list_batch)

    if num_rout > 1:
        b_inst_range = get_range_io_routing(b_inst_list_batch)
        b_new_range = get_range_io_routing(b_new_list_batch)
        b_old_range = get_range_io_routing(b_old_list_batch)
    else:
        b_inst_range = None
        b_new_range =  None
        b_old_range = None

    return input_range, output_range, input_hat_range, output_ns_range, output_s_range, cc_range, b_inst_range, b_new_range, b_old_range


def get_range_io(batch):
    temp_min = np.min(batch)
    temp_max = np.max(batch)
    range_io = {'min': temp_min, 'max': temp_max}

    return range_io


def get_range_io_routing(batch_list):
    for i, batch in enumerate(batch_list):
        temp_min = np.min(batch)
        temp_max = np.max(batch)
        if i == 0:
            range_io = {'min_' + str(i): temp_min, 'max_' + str(i): temp_max}
        else:
            range_io['min_' + str(i)] = temp_min
            range_io['max_' + str(i)] = temp_max

    return range_io


def get_act_q_format_std_layer(io_layer, optimization):
    qm, qn = get_q_format(io_layer["input"]["min"], io_layer["input"]["max"], optimization)
    qmn_input = {"int_bits": qm, "frac_bits": qn}
    qm, qn = get_q_format(io_layer["output"]["min"], io_layer["output"]["max"], optimization)
    qmn_output = {"int_bits": qm, "frac_bits": qn}
    fmt = {"layer": io_layer['layer'], "input": qmn_input, "output": qmn_output}

    return fmt


def get_act_q_format_pcap(io_layer, optimization):
    qm, qn = get_q_format(io_layer["input"]["min"], io_layer["input"]["max"], optimization)
    qmn_input = {"int_bits": qm, "frac_bits": qn}
    qm, qn = get_q_format(io_layer['output']["min"], io_layer['output']["max"], optimization)
    qmn_output = {"int_bits": qm, "frac_bits": qn}
    qm, qn = get_q_format(io_layer['output_ns']["min"], io_layer['output_ns']["max"], optimization)
    qmn_output_ns = {"int_bits": qm, "frac_bits": qn}
    fmt = {"layer": io_layer['layer'], "input": qmn_input, "output_ns": qmn_output_ns, "output": qmn_output}

    return fmt


def get_act_q_format_cap(io_layer, num_rout, optimization):
    qm, qn = get_q_format(io_layer["input"]["min"], io_layer["input"]["max"], optimization)
    qmn_input = {"int_bits": qm, "frac_bits": qn}
    qm, qn = get_q_format(io_layer['input_hat']["min"], io_layer['input_hat']["max"], optimization)
    qmn_input_hat = {"int_bits": qm, "frac_bits": qn}
    fmt = {"layer": io_layer['layer'], "input": qmn_input, "input_hat": qmn_input_hat}

    # Gets the quantized format of the output of every routing iteration
    for i in range(num_rout):
        qm, qn = get_q_format(io_layer["output_ns"]["min_"+str(i)], io_layer["output_ns"]["max_"+str(i)], optimization)
        qmn_output_ns = {"int_bits": qm, "frac_bits": qn}
        fmt["output_ns_" + str(i)] = qmn_output_ns

        qm, qn = get_q_format(io_layer["output_s"]["min_"+str(i)], io_layer["output_s"]["max_"+str(i)], optimization)
        qmn_output_s = {"int_bits": qm, "frac_bits": qn}
        fmt["output_s_" + str(i)] = qmn_output_s

        qm, qn = get_q_format(io_layer["cc"]["min_"+str(i)], io_layer["cc"]["max_"+str(i)], False)
        qmn_cc = {"int_bits": qm, "frac_bits": qn}
        fmt["cc_" + str(i)] = qmn_cc

    for i in range(num_rout-1):
        qm, qn = get_q_format(io_layer["b_inst"]["min_"+str(i)], io_layer["b_inst"]["max_"+str(i)], optimization)
        qmn_b_inst = {"int_bits": qm, "frac_bits": qn}
        fmt["b_inst_" + str(i)] = qmn_b_inst

        qm, qn = get_q_format(io_layer["b_old"]["min_"+str(i)], io_layer["b_old"]["max_"+str(i)], optimization)
        qmn_b_old = {"int_bits": qm, "frac_bits": qn}
        fmt["b_old_" + str(i)] = qmn_b_old

        qm, qn = get_q_format(io_layer["b_new"]["min_"+str(i)], io_layer["b_new"]["max_"+str(i)], optimization)
        qmn_b_new = {"int_bits": qm, "frac_bits": qn}
        fmt["b_new_" + str(i)] = qmn_b_new

    qm, qn = get_q_format(io_layer['output']["min"], io_layer['output']["max"], optimization)
    qmn_output = {"int_bits": qm, "frac_bits": qn}
    fmt["output"] = qmn_output

    return fmt


def search_dictionaries(key, value, list_of_dicts):
    for dictio in list_of_dicts:
        if dictio[key] == value:
            return dictio