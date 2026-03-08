# Guida Configurazione Eclipse CDT su Debian 12

## 1. Installazione Prerequisiti

Prima di installare Eclipse, assicurati di avere tutti gli strumenti necessari:

```bash
# Aggiorna il sistema
sudo apt update && sudo apt upgrade -y

# Installa Java Runtime Environment (richiesto da Eclipse)
sudo apt install default-jre -y

# Installa toolchain C/C++
sudo apt install build-essential -y

# Installa GCC, G++, GDB (debugger)
sudo apt install gcc g++ gdb -y

# Installa CMake (build system moderno)
sudo apt install cmake -y

# Installa Ninja (build tool veloce per CMake)
sudo apt install ninja-build -y

# Installa Git (version control)
sudo apt install git -y

# Verifica le installazioni
gcc --version
g++ --version
gdb --version
cmake --version
ninja --version
```

## 2. Installazione Eclipse CDT

### Opzione A: Tramite Package Manager (più semplice)
```bash
sudo apt install eclipse-cdt -y
```

### Opzione B: Download Manuale (versione più recente)
```bash
# Scarica Eclipse CDT dalla pagina ufficiale
# https://www.eclipse.org/downloads/packages/

# Estrai l'archivio
cd ~/Downloads
tar -xzf eclipse-cpp-*.tar.gz

# Sposta in /opt
sudo mv eclipse /opt/

# Crea link simbolico
sudo ln -s /opt/eclipse/eclipse /usr/local/bin/eclipse

# Crea desktop entry
cat > ~/.local/share/applications/eclipse.desktop << 'EOF'
[Desktop Entry]
Type=Application
Name=Eclipse CDT
Comment=Eclipse IDE for C/C++
Icon=/opt/eclipse/icon.xpm
Exec=/opt/eclipse/eclipse
Terminal=false
Categories=Development;IDE;
EOF
```

## 3. Prima Configurazione di Eclipse

### Avvio e Workspace
1. Avvia Eclipse: `eclipse` o cerca "Eclipse" nel menu applicazioni
2. Scegli una directory per il workspace (es: `~/eclipse-workspace`)
3. Seleziona "Use this as the default and do not ask again" se preferisci

### Configurazione Base
1. **Window → Preferences → General → Workspace**
   - Abilita "Refresh using native hooks or polling"
   - Abilita "Save automatically before build"

2. **Window → Preferences → General → Editors → Text Editors**
   - Abilita "Show line numbers"
   - Abilita "Show print margin" (colonna 80 o 120)

## 4. Configurazione Toolchain C/C++

### Verifica Toolchain Rilevate
1. **Window → Preferences → C/C++ → Build → Environment**
2. **Window → Preferences → C/C++ → Build → Settings**
   - Verifica che GCC sia rilevato automaticamente

### Configurazione Compilatore
1. **Window → Preferences → C/C++ → Build → Settings → Discovery**
   - Seleziona "CDT GCC Built-in Compiler Settings"

2. **Window → Preferences → C/C++ → Code Analysis**
   - Configura le regole di analisi statica secondo preferenze

## 5. Configurazione Cross-Platform Toolchain

### Per Compilazione Multi-Architettura

```bash
# Installa cross-compiler per ARM (esempio)
sudo apt install gcc-arm-linux-gnueabihf g++-arm-linux-gnueabihf -y

# Installa cross-compiler per Windows (MinGW)
sudo apt install mingw-w64 -y

# Verifica installazioni
arm-linux-gnueabihf-gcc --version
x86_64-w64-mingw32-gcc --version
```

### Configurazione in Eclipse

1. **Window → Preferences → C/C++ → Build → Settings → Tool Chains**
2. Click su "Add..." per aggiungere nuove toolchain
3. Configura per ogni target:
   - **Nome**: ARM Linux, Windows MinGW, etc.
   - **Compiler**: percorso al cross-compiler
   - **Linker**: percorso al cross-linker

## 6. Creazione Primo Progetto C++

### Nuovo Progetto
1. **File → New → C++ Project**
2. Scegli:
   - **Project name**: HelloWorld
   - **Project type**: Executable → Hello World C++ Project
   - **Toolchains**: Linux GCC (o altra toolchain)
3. Click "Finish"

### Configurazione Build
1. Click destro sul progetto → **Properties**
2. **C/C++ Build → Settings**
   - **GCC C++ Compiler → Dialect**: ISO C++17 o C++20
   - **GCC C++ Compiler → Optimization**: -O0 (debug) o -O2 (release)
   - **GCC C++ Compiler → Warnings**: -Wall -Wextra
   - **GCC C++ Linker → Libraries**: aggiungi librerie necessarie

### Build Configurations
1. **Project → Build Configurations → Manage**
2. Crea configurazioni multiple:
   - **Debug**: ottimizzazione -O0, simboli debug -g
   - **Release**: ottimizzazione -O2 o -O3
   - **Cross-ARM**: usa toolchain ARM
   - **Cross-Windows**: usa MinGW

## 7. Configurazione Debugger (GDB)

1. **Run → Debug Configurations**
2. Click destro su "C/C++ Application" → New Configuration
3. Configura:
   - **Main → C/C++ Application**: percorso all'eseguibile
   - **Debugger → GDB debugger**: `gdb` (o percorso specifico)
   - **Debugger → GDB command file**: opzionale, per comandi custom

## 8. Plugin Utili

### Installazione Plugin
**Help → Eclipse Marketplace**

Plugin consigliati:
- **CMake Editor**: syntax highlighting per CMakeLists.txt
- **Valgrind**: memory profiling e leak detection
- **Doxygen**: generazione documentazione
- **Git Integration**: già incluso (EGit)

## 9. Configurazione CMake (Opzionale ma Consigliato)

```bash
# Installa plugin CMake per Eclipse
# Help → Install New Software
# Add repository: https://download.eclipse.org/tools/cdt/releases/latest
```

### Progetto con CMake
1. **File → New → C++ Project**
2. Scegli **CMake Project**
3. Eclipse genererà automaticamente CMakeLists.txt

## 10. Ottimizzazioni Performance

### Aumenta Memoria Eclipse
Modifica `eclipse.ini` (in `/opt/eclipse/` o nella directory di installazione):

```ini
-Xms512m
-Xmx2048m
-XX:+UseG1GC
```

### Disabilita Indicizzazione Automatica (se lenta)
**Window → Preferences → C/C++ → Indexer**
- Deseleziona "Enable indexer" temporaneamente per progetti grandi

## 11. Shortcuts Utili

| Shortcut | Azione |
|----------|--------|
| `Ctrl+Space` | Auto-completamento |
| `Ctrl+Shift+F` | Formattazione codice |
| `F3` | Vai a definizione |
| `Ctrl+Alt+H` | Call hierarchy |
| `Ctrl+O` | Quick outline |
| `Ctrl+Shift+R` | Apri risorsa |
| `Ctrl+B` | Build progetto |
| `F11` | Debug |
| `Ctrl+F11` | Run |

## 12. Template Progetto Cross-Platform

Struttura consigliata:

```
my-cpp-project/
├── src/
│   ├── main.cpp
│   └── ...
├── include/
│   └── ...
├── build/
│   ├── debug/
│   ├── release/
│   ├── arm/
│   └── windows/
├── CMakeLists.txt
└── README.md
```

## Troubleshooting Comuni

### Eclipse non trova il compilatore
```bash
# Verifica PATH
echo $PATH

# Aggiungi al .bashrc se necessario
export PATH="/usr/bin:$PATH"
```

### Errori di indicizzazione
- **Project → C/C++ Index → Rebuild**
- Oppure disabilita temporaneamente l'indexer

### Build fallisce ma compilazione manuale funziona
- Verifica variabili ambiente in Eclipse
- **Window → Preferences → C/C++ → Build → Environment**

## Risorse Aggiuntive

- [Eclipse CDT Documentation](https://eclipse.org/cdt/)
- [GCC Documentation](https://gcc.gnu.org/onlinedocs/)
- [CMake Tutorial](https://cmake.org/cmake/help/latest/guide/tutorial/)

---

**Nota**: Questa guida è stata creata per Debian 12. Per altre distribuzioni, adatta i comandi del package manager.
