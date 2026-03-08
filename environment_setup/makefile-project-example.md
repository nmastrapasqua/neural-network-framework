# Esempio Progetto Makefile Multi-File in Eclipse

## Struttura Progetto

```
MyMakefileProject/
├── Makefile
├── src/
│   ├── main.cpp
│   ├── calculator.cpp
│   └── utils.cpp
├── include/
│   ├── calculator.h
│   └── utils.h
├── obj/
│   └── (file .o generati automaticamente)
└── bin/
    └── (eseguibile finale)
```

## 1. Creazione Progetto in Eclipse

1. **File → New → C++ Project**
2. Scegli **Makefile Project → Empty Project**
3. Nome: `MyMakefileProject`
4. Toolchain: **Linux GCC**
5. Click **Finish**

## 2. Crea la Struttura Directory

Nel Project Explorer, click destro sul progetto:
- **New → Folder** → crea `src`
- **New → Folder** → crea `include`
- **New → Folder** → crea `obj`
- **New → Folder** → crea `bin`

## 3. File di Esempio

### include/calculator.h
```cpp
#ifndef CALCULATOR_H
#define CALCULATOR_H

class Calculator {
public:
    int add(int a, int b);
    int subtract(int a, int b);
    int multiply(int a, int b);
    double divide(int a, int b);
};

#endif // CALCULATOR_H
```

### include/utils.h
```cpp
#ifndef UTILS_H
#define UTILS_H

#include <string>

namespace Utils {
    void printMessage(const std::string& msg);
    int getRandomNumber(int min, int max);
}

#endif // UTILS_H
```

### src/calculator.cpp
```cpp
#include "calculator.h"
#include <stdexcept>

int Calculator::add(int a, int b) {
    return a + b;
}

int Calculator::subtract(int a, int b) {
    return a - b;
}

int Calculator::multiply(int a, int b) {
    return a * b;
}

double Calculator::divide(int a, int b) {
    if (b == 0) {
        throw std::runtime_error("Division by zero");
    }
    return static_cast<double>(a) / b;
}
```

### src/utils.cpp
```cpp
#include "utils.h"
#include <iostream>
#include <random>

namespace Utils {
    void printMessage(const std::string& msg) {
        std::cout << "[INFO] " << msg << std::endl;
    }

    int getRandomNumber(int min, int max) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(min, max);
        return dis(gen);
    }
}
```

### src/main.cpp
```cpp
#include <iostream>
#include "calculator.h"
#include "utils.h"

int main() {
    Utils::printMessage("Starting Calculator Application");
    
    Calculator calc;
    
    int a = 10, b = 5;
    
    std::cout << "a = " << a << ", b = " << b << std::endl;
    std::cout << "Addition: " << calc.add(a, b) << std::endl;
    std::cout << "Subtraction: " << calc.subtract(a, b) << std::endl;
    std::cout << "Multiplication: " << calc.multiply(a, b) << std::endl;
    std::cout << "Division: " << calc.divide(a, b) << std::endl;
    
    int random = Utils::getRandomNumber(1, 100);
    std::cout << "Random number: " << random << std::endl;
    
    Utils::printMessage("Application finished");
    
    return 0;
}
```

## 4. Makefile Completo

### Makefile (nella root del progetto)
```makefile
# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -Werror -pedantic
DEBUGFLAGS = -g -O0
RELEASEFLAGS = -O2 -DNDEBUG

# Directories
SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

# Target executable
TARGET = $(BIN_DIR)/calculator

# Source files
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(SOURCES))

# Include paths
INCLUDES = -I$(INC_DIR)

# Libraries (esempio: -lpthread -lm)
LIBS = 

# Default build mode (debug or release)
BUILD_MODE ?= debug

ifeq ($(BUILD_MODE),release)
    CXXFLAGS += $(RELEASEFLAGS)
else
    CXXFLAGS += $(DEBUGFLAGS)
endif

# Default target
all: directories $(TARGET)

# Create directories if they don't exist
directories:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

# Link object files to create executable
$(TARGET): $(OBJECTS)
	@echo "Linking: $@"
	$(CXX) $(CXXFLAGS) $(OBJECTS) -o $@ $(LIBS)
	@echo "Build complete: $@"

# Compile source files to object files
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	@echo "Compiling: $<"
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(OBJ_DIR)/*.o $(TARGET)
	@echo "Clean complete"

# Clean everything including directories
distclean: clean
	rm -rf $(OBJ_DIR) $(BIN_DIR)

# Run the program
run: $(TARGET)
	@echo "Running $(TARGET)..."
	@./$(TARGET)

# Build in release mode
release:
	$(MAKE) BUILD_MODE=release

# Build in debug mode
debug:
	$(MAKE) BUILD_MODE=debug

# Show variables (for debugging Makefile)
info:
	@echo "CXX: $(CXX)"
	@echo "CXXFLAGS: $(CXXFLAGS)"
	@echo "SOURCES: $(SOURCES)"
	@echo "OBJECTS: $(OBJECTS)"
	@echo "TARGET: $(TARGET)"
	@echo "BUILD_MODE: $(BUILD_MODE)"

# Phony targets (not actual files)
.PHONY: all clean distclean run release debug info directories
```

## 5. Configurazione Eclipse per Makefile

### A. Configurazione Build

1. Click destro sul progetto → **Properties**
2. **C/C++ Build**:
   - ✅ Deseleziona **"Generate Makefiles automatically"**
   - Build command: `make`
   - Build directory: `${workspace_loc:/MyMakefileProject}`

3. **C/C++ Build → Behavior**:
   - Build (Incremental build): `all` (o lascia vuoto)
   - Clean: `clean`

### B. Configurazione Build Configurations

Crea configurazioni multiple per Debug e Release:

1. **Project → Build Configurations → Manage...**
2. Click **New...** per creare nuove configurazioni:
   - **Debug**: `make debug`
   - **Release**: `make release`

Per ogni configurazione:
1. Click destro sul progetto → **Properties**
2. Seleziona la configurazione dal dropdown in alto
3. **C/C++ Build → Behavior**:
   - Per Debug: Build command arguments: `debug`
   - Per Release: Build command arguments: `release`

### C. Configurazione Include Paths (per Indexer)

1. Click destro sul progetto → **Properties**
2. **C/C++ General → Paths and Symbols**
3. Tab **Includes** → seleziona **GNU C++**
4. Click **Add...** → **Workspace...** → seleziona `MyMakefileProject/include`
5. ✅ Spunta **"Add to all configurations"**
6. ✅ Spunta **"Add to all languages"**
7. Apply

### D. Configurazione Compiler Settings (per Indexer)

1. **C/C++ General → Preprocessor Include Paths**
2. Tab **Providers**
3. ✅ Abilita **"CDT GCC Built-in Compiler Settings"**
4. Command to get compiler specs: `${COMMAND} ${FLAGS} -E -P -v -dD "${INPUTS}"`

### E. Configurazione Symbols/Defines

1. **C/C++ General → Paths and Symbols**
2. Tab **Symbols** → seleziona **GNU C++**
3. Click **Add...** per aggiungere define:
   - Per Debug: nessun define speciale
   - Per Release: `NDEBUG`

## 6. Build del Progetto

### Da Eclipse
- **Project → Build Project** (Ctrl+B)
- Oppure click destro → **Build Project**

### Da Terminale
```bash
cd ~/eclipse-workspace/MyMakefileProject

# Build debug (default)
make

# Build release
make release

# Clean
make clean

# Build e run
make run

# Info sul Makefile
make info
```

## 7. Configurazione Debugger (GDB)

1. **Run → Debug Configurations...**
2. Click destro su **C/C++ Application** → **New Configuration**
3. Configura:
   - **Name**: MyMakefileProject Debug
   - **Main → C/C++ Application**: `bin/calculator`
   - **Main → Project**: seleziona `MyMakefileProject`
   - **Debugger → GDB debugger**: `gdb`
   - **Debugger → GDB command file**: (lascia vuoto)
4. Apply e Debug

## 8. Makefile Avanzato - Opzioni Aggiuntive

### Aggiungere Librerie Esterne

```makefile
# Esempio: aggiungere pthread e math
LIBS = -lpthread -lm

# Esempio: aggiungere Boost
INCLUDES += -I/usr/include/boost
LIBS += -lboost_system -lboost_filesystem
```

### Dependency Tracking Automatico

Aggiungi al Makefile per rigenerare automaticamente le dipendenze:

```makefile
# Generate dependency files
DEPFLAGS = -MMD -MP
CXXFLAGS += $(DEPFLAGS)

# Include dependency files
-include $(OBJECTS:.o=.d)
```

### Cross-Compilation

```makefile
# Per ARM
ifeq ($(TARGET_ARCH),arm)
    CXX = arm-linux-gnueabihf-g++
endif

# Per Windows (MinGW)
ifeq ($(TARGET_ARCH),windows)
    CXX = x86_64-w64-mingw32-g++
    TARGET = $(BIN_DIR)/calculator.exe
endif
```

Build cross-platform:
```bash
make TARGET_ARCH=arm
make TARGET_ARCH=windows
```

## 9. Comandi Make Utili

```bash
# Build normale
make

# Build verbose (mostra comandi completi)
make VERBOSE=1

# Build parallelo (usa 4 core)
make -j4

# Build con variabili custom
make CXX=clang++ CXXFLAGS="-std=c++20 -O3"

# Rebuild completo
make clean && make

# Build release e run
make release && make run
```

## 10. Troubleshooting

### Eclipse non trova gli header
- Verifica che gli include paths siano configurati in **Paths and Symbols**
- Ricostruisci l'indice: Click destro → **Index → Rebuild**

### Makefile non viene eseguito
- Verifica che **"Generate Makefiles automatically"** sia DESELEZIONATO
- Verifica il Build directory nelle proprietà

### Errori di linking
- Controlla la variabile `LIBS` nel Makefile
- Verifica che le librerie siano installate: `ldconfig -p | grep nome_lib`

### Build lenta
- Usa build parallelo: `make -j$(nproc)`
- Aggiungi nel Makefile: `MAKEFLAGS += -j$(nproc)`

## 11. Best Practices

1. **Separazione src/include**: mantieni header e implementazione separati
2. **Build artifacts fuori da src**: usa directory `obj/` e `bin/`
3. **Configurazioni multiple**: Debug (con -g) e Release (con -O2)
4. **Dependency tracking**: usa `-MMD -MP` per rigenerare dipendenze
5. **Phony targets**: dichiara sempre `.PHONY` per target non-file
6. **Variabili**: usa variabili per rendere il Makefile manutenibile
7. **Cross-platform**: testa su diverse architetture se necessario

---

Questo esempio ti fornisce una base solida per progetti Makefile professionali in Eclipse!
