# Neural Network Framework Makefile
# Compatible with Eclipse CDT and traditional make workflows

# Compiler and standard
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror -Iinclude
CXXFLAGS_SHARED = $(CXXFLAGS) -fPIC

# Library configuration
LIB_NAME = neuralnet
LIB_VERSION = 1.0.0
LIB_SONAME = lib$(LIB_NAME).so.1
LIB_SHARED = lib$(LIB_NAME).so.$(LIB_VERSION)
LIB_STATIC = lib$(LIB_NAME).a

# Installation directories
PREFIX ?= /usr/local
LIBDIR = $(PREFIX)/lib
INCLUDEDIR = $(PREFIX)/include/$(LIB_NAME)

# RapidCheck configuration
RAPIDCHECK_INCLUDE = -I/usr/local/include
RAPIDCHECK_LIB = -L/usr/local/lib -lrapidcheck -lgtest -lgtest_main -lpthread

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
OBJ_SHARED_DIR = obj/shared
BIN_DIR = bin
LIB_DIR = lib
EXAMPLE_DIR = examples
TEST_DIR = tests

# Automatic source file discovery
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_DIR)/%.o)
OBJECTS_SHARED = $(SOURCES:$(SRC_DIR)/%.cpp=$(OBJ_SHARED_DIR)/%.o)

# Header files for installation
HEADERS = $(wildcard $(INC_DIR)/*.h)

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

# Library targets
lib: $(LIB_DIR)/$(LIB_SHARED) $(LIB_DIR)/$(LIB_STATIC)

# Shared library
$(LIB_DIR)/$(LIB_SHARED): $(OBJECTS_SHARED)
	@mkdir -p $(LIB_DIR)
	@echo "Building shared library $(LIB_SHARED)..."
	$(CXX) -shared -Wl,-soname,$(LIB_SONAME) -o $@ $^
	@cd $(LIB_DIR) && ln -sf $(LIB_SHARED) $(LIB_SONAME)
	@cd $(LIB_DIR) && ln -sf $(LIB_SONAME) lib$(LIB_NAME).so
	@echo "Shared library created: $@"

# Static library
$(LIB_DIR)/$(LIB_STATIC): $(OBJECTS)
	@mkdir -p $(LIB_DIR)
	@echo "Building static library $(LIB_STATIC)..."
	ar rcs $@ $^
	@echo "Static library created: $@"

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

# Pattern rule for shared object files (with -fPIC)
$(OBJ_SHARED_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_SHARED_DIR)
	$(CXX) $(CXXFLAGS_SHARED) -c $< -o $@

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

# Install library and headers
install: lib
	@echo "Installing library to $(LIBDIR)..."
	@mkdir -p $(LIBDIR)
	@mkdir -p $(INCLUDEDIR)
	@cp $(LIB_DIR)/$(LIB_SHARED) $(LIBDIR)/
	@cp $(LIB_DIR)/$(LIB_STATIC) $(LIBDIR)/
	@rm -f $(LIBDIR)/$(LIB_SONAME) $(LIBDIR)/lib$(LIB_NAME).so
	@cd $(LIBDIR) && ln -sf $(LIB_SHARED) $(LIB_SONAME)
	@cd $(LIBDIR) && ln -sf $(LIB_SONAME) lib$(LIB_NAME).so
	@echo "Installing headers to $(INCLUDEDIR)..."
	@cp $(HEADERS) $(INCLUDEDIR)/
	@echo "Running ldconfig..."
	@ldconfig 2>/dev/null || true
	@echo ""
	@echo "Installation complete!"
	@echo "Library installed to: $(LIBDIR)"
	@echo "Headers installed to: $(INCLUDEDIR)"
	@echo ""
	@echo "To use in your projects:"
	@echo "  Compile: g++ -std=c++17 -I$(INCLUDEDIR) myapp.cpp -l$(LIB_NAME) -o myapp"
	@echo "  Or with pkg-config (if configured): g++ -std=c++17 \$$(pkg-config --cflags --libs $(LIB_NAME)) myapp.cpp -o myapp"

# Uninstall library and headers
uninstall:
	@echo "Uninstalling library from $(LIBDIR)..."
	@rm -f $(LIBDIR)/$(LIB_SHARED)
	@rm -f $(LIBDIR)/$(LIB_SONAME)
	@rm -f $(LIBDIR)/lib$(LIB_NAME).so
	@rm -f $(LIBDIR)/$(LIB_STATIC)
	@echo "Uninstalling headers from $(INCLUDEDIR)..."
	@rm -rf $(INCLUDEDIR)
	@echo "Running ldconfig..."
	@ldconfig 2>/dev/null || true
	@echo "Uninstallation complete!"

# Clean build artifacts
clean:
	rm -rf $(OBJ_DIR) $(OBJ_SHARED_DIR) $(BIN_DIR) $(LIB_DIR)

# Clean and rebuild
rebuild: clean all

# Help target
help:
	@echo "Neural Network Framework - Build Targets"
	@echo "========================================"
	@echo ""
	@echo "Build Targets:"
	@echo "  all          : Build all examples (default)"
	@echo "  lib          : Build shared and static libraries"
	@echo "  debug        : Build with debug symbols (-g -O0)"
	@echo "  release      : Build with optimizations (-O3)"
	@echo "  examples     : Build all example programs"
	@echo "  tests        : Build all unit test programs"
	@echo "  prop_tests   : Build all property test programs"
	@echo ""
	@echo "Library Targets:"
	@echo "  install      : Install library and headers to $(PREFIX)"
	@echo "  uninstall    : Remove installed library and headers"
	@echo ""
	@echo "Test Targets:"
	@echo "  test_unit    : Build and run unit tests"
	@echo "  test_property: Build and run property tests (RapidCheck)"
	@echo "  test         : Alias for test_unit"
	@echo "  test_prop    : Alias for test_property"
	@echo "  test_all     : Build and run all tests (unit + property)"
	@echo ""
	@echo "Example Targets:"
	@echo "  run_xor      : Run XOR example"
	@echo "  run_binary   : Run binary classification example"
	@echo "  run_save_load: Run save/load example"
	@echo "  run_mnist    : Run MNIST example"
	@echo "  run_iris     : Run Iris classification example"
	@echo "  run_sin      : Run sin(x) regression example"
	@echo "  run_all      : Run all examples"
	@echo ""
	@echo "Utility Targets:"
	@echo "  clean        : Remove all build artifacts"
	@echo "  rebuild      : Clean and rebuild"
	@echo "  help         : Show this help message"
	@echo ""
	@echo "Installation:"
	@echo "  Default prefix: $(PREFIX)"
	@echo "  Custom prefix: make install PREFIX=/custom/path"

# Phony targets
.PHONY: all lib debug release examples tests prop_tests test test_unit test_property test_prop test_all run_xor run_binary run_save_load run_mnist run_iris run_sin run_all install uninstall clean rebuild help

# Dependency tracking (optional, for incremental builds)
-include $(OBJECTS:.o=.d)

$(OBJ_DIR)/%.d: $(SRC_DIR)/%.cpp
	@mkdir -p $(OBJ_DIR)
	@$(CXX) $(CXXFLAGS) -MM -MT '$(OBJ_DIR)/$*.o' $< > $@
