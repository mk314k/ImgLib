CXX ?= g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra -Iinclude -Ithird_party

BUILD_DIR = build
LIB_TARGET = $(BUILD_DIR)/libimglib.a
CLI_TARGET = $(BUILD_DIR)/imglib
TEST_TARGET = $(BUILD_DIR)/imglib_tests

LIB_SOURCES = \
	src/canvas.cpp \
	src/io.cpp \
	src/color.cpp \
	src/transform.cpp \
	src/filter.cpp \
	src/warp.cpp \
	src/stylize.cpp \
	third_party/lodepng.cpp

LIB_OBJECTS = $(patsubst %.cpp,$(BUILD_DIR)/%.o,$(LIB_SOURCES))
CLI_OBJECT = $(BUILD_DIR)/src/cli.o
TEST_OBJECT = $(BUILD_DIR)/tests/test_imglib.o

all: $(LIB_TARGET) $(CLI_TARGET)

$(LIB_TARGET): $(LIB_OBJECTS)
	ar rcs $@ $^

$(CLI_TARGET): $(CLI_OBJECT) $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $(CLI_OBJECT) $(LIB_TARGET)

$(TEST_TARGET): $(TEST_OBJECT) $(LIB_TARGET)
	$(CXX) $(CXXFLAGS) -o $@ $(TEST_OBJECT) $(LIB_TARGET)

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c -o $@ $<

test: $(TEST_TARGET)
	./$(TEST_TARGET)

check: test

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all test check clean
