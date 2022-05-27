import os
import serial
import struct
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
    dataset_size = model_details['dataset_size']
    y_true = get_labels(model_details)
    y_pred = []

    print('-->Connect the MCU and press any key to proceed')
    c = input()
    ser = serial.Serial('COM5', 220000, parity=serial.PARITY_NONE, timeout=None)

    for sample_idx in tqdm(range(dataset_size), leave=True, position=0):
        pck = ser.read(1)
        y_pred_row = struct.unpack('B', pck)
        y_pred.append(y_pred_row)
    
    acc = metrics.accuracy_score(y_true, y_pred)
    print(' ACC: ' + str(acc))


if __name__ == '__main__':
    model_details = {'dataset_size': 100,
                    'labels_dir': os.path.dirname(os.path.abspath(__file__)) + '/logs/labels.bin'}

    main(model_details)