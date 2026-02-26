CC = gcc
CFLAGS = -Wall -g -Iinclude -mavx512f -fopenmp
CFLAGS += -O3
# CFLAGS += -lprofiler 
# CFLAGS += -fsanitize=address,undefined -O0 -g -fno-omit-frame-pointer

BUILD_DIR = build
TARGET = gradino

SRCS = main.c $(wildcard src/*.c)
LIB_SRCS = $(wildcard src/*.c)

OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)
LIB_OBJS = $(LIB_SRCS:%.c=$(BUILD_DIR)/%.o)

EXAMPLE_SRCS = $(wildcard examples/*.c)
EXAMPLE_BINS = $(EXAMPLE_SRCS:examples/%.c=examples/bin/%)

run: $(TARGET)
	CPUPROFILE=/tmp/prof.out ./$(TARGET)

all: $(TARGET) examples

examples: $(EXAMPLE_BINS)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ -lm

examples/bin/%: examples/%.c $(LIB_OBJS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -o $@ $< $(LIB_OBJS) -lm

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET) examples/bin

.PHONY: all clean examples
