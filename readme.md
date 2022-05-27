# Q7-CapsNets
## Capsule Networks (Dynamic Routing) on Arm Cortex-M and RISC-V GAP8

## Features
- C API (based on CMSIS-NN) for the execution of quantized Capsule Networks on Arm Cortex-M MCUs 
- C API (based on PULP-NN) for the execution of quantized Capsule Networks on RISC-V GAP8 MCUs 
- Python framework for the quantization of Capsule Networks developed in TensorFlow
- Python scripts to simulate the behavior of the quantized CapsNet on MCUs (fast testing of the accuracy returned by the Q7-CapsNet)
- Implementation examples for the MNIST, CIFAR-10, and SmallNORB datasets

## Tests and Results

| Dataset | Float-32 CapsNet ACC| Q7 CapsNet ACC (Arm) | Q7 CapsNet ACC (RISC-V) |
| ------ | ------ | ------ |------ |
| MNIST | 99.01% | 98.83% | 98.85% |
| CIFAR-10 | 78.54% | 78.50% | 78.38% |
| SmallNORB | 92.56% | 92.50% | 92.49% |

## Directory Structure
| Directory | Content |
| ------ | ------ |
| Arm_Cortex-M/API | C API for Arm Cortex-M |
| Arm_Cortex-M/Examples | Examples implementing Q7-CapsNets on Arm Cortex-M |
| RISC-V_GAP8/API | C API for RISC-V GAP8 |
| RISC-V_GAP8/pulp-nn | Fork of pulp-nn supporting conv2d with signed int8 |
| RISC-V_GAP8/Examples | Examples implementing Q7-CapsNets on RISC-V GAP8 |
| Quantization_Framework | Python framework to quantize a CapsNet developed in TensorFlow and saved as .h5 |
| Simulator | Examples on how to fast testing the accuracy of Q7-CapsNets |

## Help and Support
### Communication
- E-mail: miguel.costa@dei.uminho.pt

### Citations
If you use Q7-CapsNets in a scientific publication, we would appreciate citations: https://arxiv.org/abs/2110.02911
