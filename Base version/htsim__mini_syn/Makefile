# 保持原有的路径定义
HTSIM_DIR ?= ../csg-htsim/sim

CXX      := g++
# 建议加上 -D参数，有时候库需要特定的宏
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -I$(HTSIM_DIR) -I.

HTSIM_EXCLUDES := $(HTSIM_DIR)/main.o \
                  $(HTSIM_DIR)/parse_output.o

# 使用 filter-out 过滤掉这些不需要的文件
HTSIM_SRCS := $(filter-out $(HTSIM_EXCLUDES), $(wildcard $(HTSIM_DIR)/*.o))

# 本项目源文件
SIM_SRCS := main.cpp gpu_node_sync.cpp

TARGET := moe_sync.bin

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SIM_SRCS) sync_scheduler.h tofino_switch_sync.h gpu_node_sync.h
	$(CXX) $(CXXFLAGS) -o $@ $(SIM_SRCS) $(HTSIM_SRCS) 2>&1
	@echo "Built: $(TARGET)"

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o
