CC = gcc
CFLAGS = -Wall -g -Iinclude -mavx512f
# CFLAGS += -O3
# CFLAGS += -lprofiler 

BUILD_DIR = build
TARGET = gradino

SRCS = main.c $(wildcard src/*.c)

OBJS = $(SRCS:%.c=$(BUILD_DIR)/%.o)

run: $(TARGET)
	CPUPROFILE=/tmp/prof.out ./$(TARGET)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^

$(BUILD_DIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -rf $(BUILD_DIR) $(TARGET)

.PHONY: all clean
