#define DATASET_SIZE            10000

#define CONV1_IN_DIM			28
#define CONV1_CH_DIM			1
#define CONV1_NUM_FILTERS		16
#define CONV1_KERNEL_SIZE		7
#define CONV1_PADDING			0
#define	CONV1_STRIDE			1
#define CONV1_OUT_DIM			22
#define CONV1_BIAS_LSHIFT       5
#define CONV1_OUT_RSHIFT        8

#define PCAP_NUM_CAP	 		16
#define PCAP_DIM_CAP			4
#define PCAP_IN_DIM				CONV1_OUT_DIM
#define PCAP_CH_DIM				CONV1_NUM_FILTERS
#define PCAP_KERNEL_SIZE        7
#define PCAP_PADDING			0
#define	PCAP_STRIDE				2	
#define PCAP_OUT_DIM			8
#define PCAP_SQUASH_IN_QN       1
#define PCAP_SQUASH_OUT_QN      7
#define PCAP_BIAS_LSHIFT        2			
#define PCAP_OUT_RSHIFT         10

#define CAP_NUM_CAP				10
#define CAP_DIM_CAP				6
#define CAP_INPUT_NUM_CAP		PCAP_OUT_DIM * PCAP_OUT_DIM * PCAP_NUM_CAP
#define CAP_INPUT_DIM_CAP		PCAP_DIM_CAP
#define CAP_NUM_ROUT			3
#define CAP_SQUASH_IN_QN		{6,5,4}
#define CAP_SQUASH_OUT_QN		{7,7,7}
#define CAP_IN_HAT_RSHIFT		7
#define CAP_OUT_RSHIFTS			{7, 8, 9}
#define CAP_B_INST_RSHIFTS	    {8, 8}	
#define CAP_B_NEW_RSHIFTS		{7, 6}

#define INPUT_DATA_SIZE			CONV1_IN_DIM * CONV1_IN_DIM * CONV1_CH_DIM