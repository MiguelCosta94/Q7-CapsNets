# User Test
APP              = test
# App sources
APP_SRCS         = $(wildcard Src/*.c) $(wildcard pulp-nn/src/*.c) $(wildcard Caps_Net_Lib/Src/*.c)
# App includes
APP_INC	         =
# Compiler flags
APP_CFLAGS       = -DNUM_CORES=8 -O1 -g -IInc -Ipulp-nn/include -ICaps_Net_Lib/Inc -w
# Linker flags
APP_LDFLAGS      = -lm

FLASH_FILE_0 =	data.bin
FLASH_FILE_1 =	labels.bin
FLASH_FILES		+= $(FLASH_FILE_0) $(FLASH_FILE_1)

# list of files to include in the ReadFS.
READFS_FILES := $(FLASH_FILES)

PLPBRIDGE_FLAGS += -f

# Custom linker
APP_LINK_SCRIPT  =

include $(RULES_DIR)/pmsis_rules.mk