/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * <h2><center>&copy; Copyright (c) 2022 STMicroelectronics.
  * All rights reserved.</center></h2>
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "fatfs.h"
#include "usb_host.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "File_Handling.h"
#include "arm_nnfunctions.h"
#include "params_snorb.h"
#include "wt_q.h"
#include "bias_q.h"
#include "caps_layer.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef hlpuart1;

/* USER CODE BEGIN PV */
q7_t conv1_wt[CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE * CONV1_CH_DIM * CONV1_NUM_FILTERS] = CONV2D_WT;
q7_t conv1_bias[CONV1_NUM_FILTERS] = CONV2D_BIAS;

q7_t pcap_wt[PCAP_KERNEL_SIZE * PCAP_KERNEL_SIZE * PCAP_CH_DIM * PCAP_NUM_CAP * PCAP_DIM_CAP] = PRIMARY_CAPSULE_WT;
q7_t pcap_bias[PCAP_NUM_CAP * PCAP_DIM_CAP] = PRIMARY_CAPSULE_BIAS;

q7_t cap_wt[CAP_NUM_CAP * CAP_INPUT_NUM_CAP * CAP_DIM_CAP * CAP_INPUT_DIM_CAP] = CAPSULE_WT;
uint16_t cap_out_rshifts[CAP_NUM_ROUT] = CAP_OUT_RSHIFTS;
uint16_t cap_squash_in_qn[CAP_NUM_ROUT] = CAP_SQUASH_IN_QN;
uint16_t cap_squash_out_qn[CAP_NUM_ROUT] = CAP_SQUASH_OUT_QN;
uint16_t b_inst_rshifts[CAP_NUM_ROUT-1] = CAP_B_INST_RSHIFTS;
uint16_t b_new_rshifts[CAP_NUM_ROUT-1] = CAP_B_NEW_RSHIFTS;

q7_t out_buffer1[CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_NUM_FILTERS];
q7_t out_buffer2[CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_NUM_FILTERS];
q7_t buffer_aux[CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_NUM_FILTERS];

q7_t buffer_input_hat[CAP_NUM_CAP * CAP_INPUT_NUM_CAP * CAP_DIM_CAP];
q7_t buffer_b[CAP_NUM_CAP * CAP_INPUT_NUM_CAP];
q7_t buffer_c[CAP_NUM_CAP * CAP_INPUT_NUM_CAP];

FIL USBHDataFile;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_LPUART1_UART_Init(void);
void MX_USB_HOST_Process(void);

/* USER CODE BEGIN PFP */
q7_t run_ann(q7_t *input);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_FATFS_Init();
  MX_USB_HOST_Init();
  MX_LPUART1_UART_Init();
  /* USER CODE BEGIN 2 */
	FRESULT status_data_file;
  q7_t ann_out;

  while(get_usb_status() == 0){
		MX_USB_HOST_Process();
	}

  status_data_file = Open_File("snorb.bin", &USBHDataFile);
	if(status_data_file != FR_OK){
		return 0;
	}

  for(uint16_t i=0; i<DATASET_SIZE; i++) {
    status_data_file = Read_File_Batch("snorb.bin", &USBHDataFile, INPUT_DATA_SIZE, (char*)out_buffer2);
    if(status_data_file != FR_OK){
      break;
    }

    ann_out = run_ann(out_buffer2);
    if(HAL_UART_Transmit(&hlpuart1, (uint8_t*)&ann_out, 1, 0xFFFFFFFF) != HAL_OK) {
    }
	}
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */
    MX_USB_HOST_Process();

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};
  RCC_PeriphCLKInitTypeDef PeriphClkInit = {0};

  /** Configure the main internal regulator output voltage
  */
  if (HAL_PWREx_ControlVoltageScaling(PWR_REGULATOR_VOLTAGE_SCALE1_BOOST) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI48|RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSI48State = RCC_HSI48_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 2;
  RCC_OscInitStruct.PLL.PLLN = 30;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = RCC_PLLQ_DIV2;
  RCC_OscInitStruct.PLL.PLLR = RCC_PLLR_DIV2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }
  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
  PeriphClkInit.PeriphClockSelection = RCC_PERIPHCLK_LPUART1|RCC_PERIPHCLK_USB;
  PeriphClkInit.Lpuart1ClockSelection = RCC_LPUART1CLKSOURCE_PCLK1;
  PeriphClkInit.UsbClockSelection = RCC_USBCLKSOURCE_HSI48;
  if (HAL_RCCEx_PeriphCLKConfig(&PeriphClkInit) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief LPUART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_LPUART1_UART_Init(void)
{

  /* USER CODE BEGIN LPUART1_Init 0 */

  /* USER CODE END LPUART1_Init 0 */

  /* USER CODE BEGIN LPUART1_Init 1 */

  /* USER CODE END LPUART1_Init 1 */
  hlpuart1.Instance = LPUART1;
  hlpuart1.Init.BaudRate = 220000;
  hlpuart1.Init.WordLength = UART_WORDLENGTH_8B;
  hlpuart1.Init.StopBits = UART_STOPBITS_1;
  hlpuart1.Init.Parity = UART_PARITY_NONE;
  hlpuart1.Init.Mode = UART_MODE_TX_RX;
  hlpuart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  hlpuart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  hlpuart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  hlpuart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  hlpuart1.FifoMode = UART_FIFOMODE_DISABLE;
  if (HAL_UART_Init(&hlpuart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&hlpuart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&hlpuart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&hlpuart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN LPUART1_Init 2 */

  /* USER CODE END LPUART1_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOG_CLK_ENABLE();
  HAL_PWREx_EnableVddIO2();
  __HAL_RCC_GPIOA_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOG, GPIO_PIN_6, GPIO_PIN_RESET);

  /*Configure GPIO pin : PG6 */
  GPIO_InitStruct.Pin = GPIO_PIN_6;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);

}

/* USER CODE BEGIN 4 */
q7_t run_ann(q7_t *input)
{
	q7_t prediction = 0;
	
	// Layer 1: Just a conventional Conv2D layer
	arm_convolve_HWC_q7_basic(input, CONV1_IN_DIM, CONV1_CH_DIM, conv1_wt, CONV1_NUM_FILTERS, CONV1_KERNEL_SIZE, CONV1_PADDING,
														CONV1_STRIDE, conv1_bias, CONV1_BIAS_LSHIFT, CONV1_OUT_RSHIFT, out_buffer1, CONV1_OUT_DIM, (q15_t*)buffer_aux, NULL);
	
	arm_relu_q7(out_buffer1, CONV1_OUT_DIM * CONV1_OUT_DIM * CONV1_NUM_FILTERS);
	
	// Layer 2: Conv2D layer with `squash` activation
	primary_capsule_layer_q7_basic(out_buffer1, PCAP_IN_DIM, PCAP_IN_DIM, PCAP_CH_DIM, pcap_wt, PCAP_NUM_CAP, PCAP_DIM_CAP, 
																PCAP_KERNEL_SIZE, PCAP_KERNEL_SIZE, PCAP_PADDING, PCAP_PADDING, PCAP_STRIDE, PCAP_STRIDE, pcap_bias, PCAP_BIAS_LSHIFT,
																PCAP_OUT_RSHIFT, PCAP_SQUASH_IN_QN, PCAP_SQUASH_OUT_QN, out_buffer2, PCAP_OUT_DIM, PCAP_OUT_DIM, (q15_t*)buffer_aux);
	
	// Layer 3: Capsule layer. Routing algorithm works here
	capsule_layer_q7(out_buffer2, CAP_NUM_CAP, CAP_DIM_CAP, CAP_INPUT_NUM_CAP, CAP_INPUT_DIM_CAP, CAP_NUM_ROUT, cap_wt, CAP_IN_HAT_RSHIFT, cap_out_rshifts,
										b_inst_rshifts, b_new_rshifts, cap_squash_in_qn, cap_squash_out_qn, out_buffer1, buffer_input_hat, buffer_b, buffer_c, buffer_aux);
	
	// Layer 4: Length layer
	capsule_length_q7(out_buffer1, CAP_NUM_CAP, CAP_DIM_CAP, out_buffer2);
	
	for(uint16_t i=0; i<CAP_NUM_CAP; i++){
		if(out_buffer2[i] > out_buffer2[prediction]){
			prediction = i;
		}
	}
	
	return prediction;
}
/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */

/************************ (C) COPYRIGHT STMicroelectronics *****END OF FILE****/
