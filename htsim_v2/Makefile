# 保持原有的路径定义
HTSIM_DIR ?= ./csg-htsim/sim

CXX      := g++
# 建议加上 -D参数，有时候库需要特定的宏
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -I$(HTSIM_DIR) -I.

HTSIM_EXCLUDES := $(HTSIM_DIR)/main.o \
                  $(HTSIM_DIR)/parse_output.o

# 使用 filter-out 过滤掉这些不需要的文件
HTSIM_OBJS := $(filter-out $(HTSIM_EXCLUDES), $(wildcard $(HTSIM_DIR)/*.o))

# 本项目源文件
SIM_SRCS := main.cpp gpu_node.cpp
SIM_OBJS := $(SIM_SRCS:.cpp=.o)

TARGET := moe_sim.bin

.PHONY: all run clean lib

all: lib $(TARGET)

# 确保基础库先被编译
lib:
	$(MAKE) -C $(HTSIM_DIR)

# 链接时包含库的 .o 文件
$(TARGET): $(SIM_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(HTSIM_OBJS)
	@echo "Built: $(TARGET)"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o
	# 如果想彻底清理库：$(MAKE) -C $(HTSIM_DIR) clean