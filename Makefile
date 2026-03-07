# Neural Network Framework Makefile
# Compatible with Eclipse CDT and traditional make workflows

# Compiler and standard
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror -Iinclude

# RapidCheck configuration
RAPIDCHECK_INCLUDE = -I/usr/local/include
RAPIDCHECK_LIB = -L/usr/local/lib -lrapidcheck -lgtest -lgtest_main -lpthread

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin
EXAMPLE_DIR = examples
TEST_DIR = tests

# Automatic source file discovery
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)

# Example sources
EXAMPLE_SOURCES = $(wildcard $(EXAMPLE_DIR)/*.cpp)
EXAMPLE_BINS = $(EXAMPLE_SOURCES:$(EXAMPLE_DIR)/%.cpp=$(BIN_DIR)/%)

# Test sources
TEST_SOURCES = $(wildcard $(TEST_DIR)/test_*.cpp)
TEST_BINS = $(TEST_SOURCES:$(TEST_DIR)/%.cpp=$(BIN_DIR)/%)

# Property test sources
PROP_SOURCES = $(wildcard $(TEST_DIR)/prop_*.cpp)
PROP_BINS = $(PROP_SOURCES:$(TEST_DIR)/%.cpp=$(BIN_DIR)/%)

# Default target
all: examples

# Configuration targets
debug: CXXFLAGS += -g -O0 -DDEBUG
debug: clean all

release: CXXFLAGS += -O3 -DNDEBUG
release: clean all

# Build examples
examples: $(EXAMPLE_BINS)

# Build tests
tests: $(TEST_BINS)

# Build property tests
prop_tests: $(PROP_BINS)

# Pattern rule for object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Pattern rule for example executables
$(BIN_DIR)/%: $(EXAMPLE_DIR)/%.cpp $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Pattern rule for test executables
$(BIN_DIR)/test_%: $(TEST_DIR)/test_%.cpp $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $^ -o $@

# Pattern rule for property test executables (with RapidCheck)
$(BIN_DIR)/prop_%: $(TEST_DIR)/prop_%.cpp $(OBJECTS)
	@mkdir -p $(BIN_DIR)
	$(CXX) $(CXXFLAGS) $(RAPIDCHECK_INCLUDE) $^ -o $@ $(RAPIDCHECK_LIB)

# Run examples
run_xor: $(BIN_DIR)/xor_example
	@echo "Running XOR example..."
	@$(BIN_DIR)/xor_example

run_binary: $(BIN_DIR)/binary_classification
	@echo "Running binary classification example..."
	@$(BIN_DIR)/binary_classification

run_save_load: $(BIN_DIR)/save_load_example
	@echo "Running save/load example..."
	@$(BIN_DIR)/save_load_example

run_mnist: $(BIN_DIR)/mnist_example
	@echo "Running MNIST example..."
	@$(BIN_DIR)/mnist_example

run_iris: $(BIN_DIR)/iris_example
	@echo "Running Iris example..."
	@$(BIN_DIR)/iris_example

run_sin: $(BIN_DIR)/sin_regression_example
	@echo "Running sin(x) regression example..."
	@$(BIN_DIR)/sin_regression_example

# Run all examples
run_all: run_xor run_binary run_save_load run_mnist run_iris run_sin

# Run tests
test: tests
	@echo "Running unit tests..."
	@for test in $(TEST_BINS); do \
		echo "Running $$test..."; \
		$$test || exit 1; \
	done

# Run property tests
test_prop: prop_tests
	@echo "Running property tests..."
	@for test in $(PROP_BINS); do \
		echo "Running $$test..."; \
		$$test --rc-params="max_success=100" || exit 1; \
	done

# Run all tests (unit + property)
test_all: test test_prop

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Clean and rebuild
rebuild: clean all

# Help target
help:
	@echo "Neural Network Framework - Build Targets"
	@echo "========================================"
	@echo "all          : Build all examples (default)"
	@echo "debug        : Build with debug symbols (-g -O0)"
	@echo "release      : Build with optimizations (-O3)"
	@echo "examples     : Build all example programs"
	@echo "tests        : Build all unit test programs"
	@echo "prop_tests   : Build all property test programs"
	@echo "test_unit    : Build and run unit tests"
	@echo "test_property: Build and run property tests (RapidCheck)"
	@echo "test         : Alias for test_unit"
	@echo "test_prop    : Alias for test_property"
	@echo "test_all     : Build and run all tests (unit + property)"
	@echo "run_xor      : Run XOR example"
	@echo "run_binary   : Run binary classification example"
	@echo "run_save_load: Run save/load example"
	@echo "run_mnist    : Run MNIST example"
	@echo "run_iris     : Run Iris classification example"
	@echo "run_sin      : Run sin(x) regression example"
	@echo "run_all      : Run all examples"
	@echo "clean        : Remove all build artifacts"
	@echo "rebuild      : Clean and rebuild"
	@echo "help         : Show this help message"

# Phony targets
.PHONY: all debug release examples tests prop_tests test test_unit test_property test_prop test_all run_xor run_binary run_save_load run_mnist run_iris run_sin run_all clean rebuild help

# Dependency tracking (optional, for incremental builds)
-include $(OBJECTS:.o=.d)

$(OBJ_DIR)/%.d: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	@$(CXX) $(CXXFLAGS) -MM -MT '$(OBJ_DIR)/$*.o' $< > $@
