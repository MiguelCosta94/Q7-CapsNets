#include "pmsis.h"
#include "bsp/fs.h"
#include "bsp/fs/readfs.h"
#include "bsp/flash/hyperflash.h"
#include "stdio.h"
#include "pulp_nn_kernels.h"
#include "caps_layer.h"
#include "params_snorb.h"
#include "wt_q.h"
#include "bias_q.h"

/* -------------------------------PRIVATE VARIABLES --------------------------------------------*/
int8_t conv1_wt[CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE * CONV1_CH_DIM * CONV1_NUM_FILTERS] = CONV2D_WT;
int8_t conv1_bias[CONV1_NUM_FILTERS] = CONV2D_BIAS;

int8_t pcap_wt[PCAP_KERNEL_SIZE * PCAP_KERNEL_SIZE * PCAP_CH_DIM * PCAP_NUM_CAP * PCAP_DIM_CAP] = PRIMARY_CAPSULE_WT;
int8_t pcap_bias[PCAP_NUM_CAP * PCAP_DIM_CAP] = PRIMARY_CAPSULE_BIAS;

int8_t cap_wt[CAP_NUM_CAP * CAP_INPUT_NUM_CAP * CAP_DIM_CAP * CAP_INPUT_DIM_CAP] = CAPSULE_WT;
uint16_t cap_out_rshifts[CAP_NUM_ROUT] = CAP_OUT_RSHIFTS;
uint16_t cap_squash_in_qn[CAP_NUM_ROUT] = CAP_SQUASH_IN_QN;
uint16_t cap_squash_out_qn[CAP_NUM_ROUT] = CAP_SQUASH_OUT_QN;
uint16_t b_inst_rshifts[CAP_NUM_ROUT-1] = CAP_B_INST_RSHIFTS;
uint16_t b_new_rshifts[CAP_NUM_ROUT-1] = CAP_B_NEW_RSHIFTS;

int8_t out_buffer1[CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_NUM_FILTERS];
int8_t out_buffer2[PCAP_OUT_DIM * PCAP_OUT_DIM * PCAP_NUM_CAP * PCAP_DIM_CAP];

int8_t buffer_aux[2 * 8 * PCAP_CH_DIM * PCAP_KERNEL_SIZE * PCAP_KERNEL_SIZE];

int8_t buffer_input_hat[CAP_NUM_CAP * CAP_INPUT_NUM_CAP * CAP_DIM_CAP];
int8_t buffer_b[CAP_NUM_CAP * CAP_INPUT_NUM_CAP];
int8_t buffer_c[CAP_NUM_CAP * CAP_INPUT_NUM_CAP];

struct files {pi_fs_file_t *data; pi_fs_file_t *labels};
/* -------------------------- PRIVATE FUNCTION PROTOTYPES -------------------------------------*/
void open_filesystem(struct pi_device *flash, struct pi_device *fs);
int8_t run_ann(void *arg);

int8_t run_ann(void *arg)
{
	int8_t *input = (int8_t*) arg;
	
	// Layer 1: Conv2D layer
	pulp_nn_conv_Ho_parallel_int8(input, CONV1_IN_DIM, CONV1_IN_DIM, CONV1_CH_DIM, conv1_wt, CONV1_NUM_FILTERS, CONV1_KERNEL_SIZE, 
						CONV1_KERNEL_SIZE, CONV1_PADDING, CONV1_PADDING, CONV1_PADDING, CONV1_PADDING, CONV1_STRIDE, CONV1_STRIDE,
						conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, 1, out_buffer1, CONV1_OUT_DIM, CONV1_OUT_DIM, NULL, NULL,
						buffer_aux, 1, 0, NULL);
	
	// Layer 2: Conv2D layer
	primary_capsule_layer_Ho_parallel_q7(out_buffer1, PCAP_IN_DIM, PCAP_IN_DIM, PCAP_CH_DIM, pcap_wt, PCAP_NUM_CAP, PCAP_DIM_CAP,
								PCAP_KERNEL_SIZE, PCAP_KERNEL_SIZE, PCAP_PADDING, PCAP_PADDING, PCAP_PADDING, PCAP_PADDING,
								PCAP_STRIDE, PCAP_STRIDE, pcap_bias, PCAP_BIAS_LSHIFT, PCAP_OUT_RSHIFT, PCAP_SQUASH_IN_QN, PCAP_SQUASH_OUT_QN,
								out_buffer2, PCAP_OUT_DIM, PCAP_OUT_DIM, buffer_aux);

	// Layer 3: Capsule layer. Routing algorithm works here
	capsule_layer_q7(out_buffer2, CAP_NUM_CAP, CAP_DIM_CAP, CAP_INPUT_NUM_CAP, CAP_INPUT_DIM_CAP, CAP_NUM_ROUT, cap_wt,
					CAP_IN_HAT_RSHIFT, cap_out_rshifts, b_inst_rshifts, b_new_rshifts, cap_squash_in_qn, cap_squash_out_qn,
					out_buffer1, buffer_input_hat, buffer_b, buffer_c, buffer_aux);
	
	// Layer 4:
	capsule_length_q7(out_buffer1, CAP_NUM_CAP, CAP_DIM_CAP, out_buffer2);
}

void open_filesystem(struct pi_device *flash, struct pi_device *fs)
{
    struct pi_readfs_conf conf;
    struct pi_hyperflash_conf flash_conf;

    /* Init & open flash. */
    pi_hyperflash_conf_init(&flash_conf);
    pi_open_from_conf(flash, &flash_conf);
    if (pi_flash_open(flash))
    {
        printf("Error flash open !\n");
        pmsis_exit(-1);
    }

    /* Open filesystem on flash. */
    pi_readfs_conf_init(&conf);
    conf.fs.flash = flash;
    pi_open_from_conf(fs, &conf);
    if (pi_fs_mount(fs))
    {
        printf("Error FS mounting !\n");
        pmsis_exit(-2);
    }
}

/* Cluster main entry, executed by core 0. */
void cluster_delegate(void *arg)
{
	printf("Cluster master core entry\n");

	struct files *f = arg;
	pi_fs_file_t *data_file = f->data;
	pi_fs_file_t *labels_file = f->labels;
	pi_cl_fs_req_t req_data, req_label;
	int32_t size = 0;
	volatile uint16_t accurate_count = 0;
	volatile uint8_t prediction = 0;
	volatile uint8_t y_true;

	printf("Starting inference...\n");

	for(uint16_t i=0; i<DATASET_SIZE; i++) {
		printf("%d\n", i);

		pi_cl_fs_read(data_file, out_buffer2, INPUT_DATA_SIZE, &req_data);
		size = pi_cl_fs_wait(&req_data);

		pi_cl_fs_read(labels_file, &y_true, 1, &req_label);
		size = pi_cl_fs_wait(&req_label);

		pi_cl_team_fork(NUM_CORES, run_ann, out_buffer2);

		for(uint16_t j=0; j<CAP_NUM_CAP; j++){
			if(out_buffer2[j] > out_buffer2[prediction]){
				prediction = j;
			}
		}

		if(prediction == y_true) {
			accurate_count++;
		}
	}

	printf("Accurate Count: %d\n", accurate_count);
	printf("ACC: %.2f\n", (float)accurate_count/DATASET_SIZE);

	pi_fs_close(data_file);
	pi_fs_close(labels_file);
}

int main()
{
	struct pi_device cluster_dev = {0};
	struct pi_cluster_conf cluster_conf;
	struct pi_device fs;
    struct pi_device flash;

	/* Init cluster configuration structure. */
    pi_cluster_conf_init(&cluster_conf);
    cluster_conf.id = 0;               /* Set cluster ID. */

	/* Configure & open cluster. */
    pi_open_from_conf(&cluster_dev, &cluster_conf);
    if (pi_cluster_open(&cluster_dev))
    {
        printf("Cluster open failed !\n");
        return -1;
    }

	/* Configure filesytem and open files*/
	open_filesystem(&flash, &fs);

	pi_fs_file_t *data_file = pi_fs_open(&fs, "data2.bin", 0);
	if (data_file == NULL)
	{
		printf("File data.bin open failed!\n");
		pmsis_exit(-3);
	}
	printf("File data.bin open success.\n");
	printf("File data.bin size: %d bytes.\n", data_file->size);

	pi_fs_file_t *labels_file = pi_fs_open(&fs, "labels2.bin", 0);
	if (labels_file == NULL)
	{
		printf("File labels.bin open failed!\n");
		pmsis_exit(-4);
	}
	printf("File labels.bin open success.\n");
	printf("File labels.bin size: %d bytes.\n", labels_file->size);

	/* Prepare cluster task and send it to cluster. */
	struct files f = {.data=data_file, .labels=labels_file};
	struct pi_cluster_task cluster_task = {0};
	cluster_task.entry = cluster_delegate;
    cluster_task.arg = &f;

	pi_cluster_send_task_to_cl(&cluster_dev, &cluster_task);
    pi_cluster_close(&cluster_dev);

	return 0;
}