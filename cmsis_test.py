import os
import serial
import struct
import numpy as np
from sklearn import metrics
from tqdm import tqdm


def get_labels(model_details):
    labels_list = []

    file = open(model_details['labels_dir'], "rb")
    labels_byte = file.read(model_details['dataset_size'])

    for label in labels_byte:
        labels_list.append(label)

    return labels_list


def main(model_details):
    model_id = model_details['id']
    dataset_size = model_details['dataset_size']
    batch_size = model_details['batch_size']
    y_true = get_labels(model_details)
    y_pred = []

    print('-->Connect the MCU and press any key to proceed')
    c = input()

    ser = serial.Serial('COM5', 220000, parity=serial.PARITY_NONE, timeout=None)

    if model_id == 'ano':
        for file_idx in tqdm(range(dataset_size), leave=True, position=0):
            y_pred_wav_file = []
            
            for n in range(batch_size):
                pck = ser.read(4)
                y_pred_row = struct.unpack('f', pck)
                y_pred_wav_file.append(y_pred_row)

            y_pred.append(np.mean(y_pred_wav_file))
    
        auc = metrics.roc_auc_score(y_true, y_pred)
        print('Dataset: ' + str(model_id) + ' AUC: ' + str(auc))

    else:
        for sample_idx in tqdm(range(dataset_size), leave=True, position=0):
            pck = ser.read(1)
            y_pred_row = struct.unpack('B', pck)
            y_pred.append(y_pred_row)
        
        acc = metrics.accuracy_score(y_true, y_pred)
        print('Dataset: ' + str(model_id) + ' ACC: ' + str(acc))



if __name__ == '__main__':
    #model_details = {'id': 'kws', 'dataset_size': 100, 'batch_size': 1, 'output_size': 12,
    #                'labels_dir': os.path.dirname(os.path.abspath(__file__)) + '/logs/labels.bin'}
    #model_details = {'id': 'vww', 'dataset_size': 100, 'batch_size': 1, 'output_size': 2,
    #                'labels_dir': os.path.dirname(os.path.abspath(__file__)) + '/logs_vww/labels.bin'}
    # id; wav files per machine id; number of input samples per wav file; output size per input sample
    #model_details = {'id': 'ano', 'dataset_size': 100, 'batch_size': 196, 'output_size': 640,
    #                'labels_dir': os.path.dirname(os.path.abspath(__file__)) + '/logs/labels.bin'}
    #model_details = {'id': 'cif', 'dataset_size': 100, 'batch_size': 1, 'output_size': 10,
    #                'labels_dir': os.path.dirname(os.path.abspath(__file__)) + '/logs/labels.bin'}
    model_details = {'id': 'sno', 'dataset_size': 100, 'batch_size': 1, 'output_size': 5,
                    'labels_dir': os.path.dirname(os.path.abspath(__file__)) + '/logs/labels.bin'}

    main(model_details)