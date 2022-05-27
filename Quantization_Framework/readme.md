# Quantization Framework
The quantization framework receives a .h5 Capsule Network developed in TensorFlow and returns a "logs" folder with the files listed below. In the input model, the name of the primary capsule layer must start with "primary_capsule". The name of capsule layers must start with "capsule".

| File | Content |
| ------ | ------ |
| act_q_format.csv | Quantization format used in the input and output of each layer and in-between computations |
| bias_q.h | Header file with the quantized bias |
| bias_q_format.csv | Quantization format used in the bias of each layer |
| bias_shift.csv | Bias shift used per layer to avoid saturation |
| data.bin | Binary file with the quantized dataset |
| labels.bin | Binary file with labels |
| output_shift.csv | Output shift(s) used per layer to avoid saturation |
| wt_q.h | Header file with the quantized weights |
| wt_q_format.csv | Quantization format used in the weights of each layer |

## Directory Structure
| File | Content |
| ------ | ------ |
| CapsuleNet_CIFAR.py | Example implementing a CapsNet for the CIFAR-10 dataset |
| CapsuleNet_MNIST.py | Example implementing a CapsNet for the MNIST dataset |
| CapsuleNet_SNORB.py | Example implementing a CapsNet for the smallNORB dataset |
| Quantization .py | Main quantization script |
| Quantization_Backend.py | Auxiliary functions for the main quantization script |
| cmsis_test.py | Script that receives predictions via serial port from Arm Cortex-M MCUs and returns the accuracy |
