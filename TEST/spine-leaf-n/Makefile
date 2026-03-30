HTSIM_DIR ?= ../csg-htsim/sim

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -I$(HTSIM_DIR) -I.

HTSIM_EXCLUDES := $(HTSIM_DIR)/main.o \
                  $(HTSIM_DIR)/parse_output.o

HTSIM_OBJS := $(filter-out $(HTSIM_EXCLUDES), $(wildcard $(HTSIM_DIR)/*.o))

SIM_SRCS := main.cpp gpu_node.cpp
SIM_OBJS := $(SIM_SRCS:.cpp=.o)

TARGET := moe_sim_512.bin

.PHONY: all run clean lib

all: lib $(TARGET)

lib:
	$(MAKE) -C $(HTSIM_DIR)

$(TARGET): $(SIM_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(HTSIM_OBJS)
	@echo "Built: $(TARGET)"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o
