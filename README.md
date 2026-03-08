# Neural Network Framework

Un framework educativo per reti neurali implementato in C++17 puro, senza dipendenze esterne (eccetto RapidCheck per i property test). Progettato per comprendere i fondamenti del deep learning attraverso un'implementazione chiara e ben documentata.

## 🎯 Scopo Educativo

Questo progetto è stato creato con finalità didattiche per:
- Comprendere l'implementazione interna delle reti neurali
- Studiare l'algoritmo di backpropagation in dettaglio
- Esplorare diverse architetture e funzioni di attivazione
- Imparare tecniche di inizializzazione dei pesi (Xavier, He)
- Sperimentare con problemi di classificazione e regressione

Il codice privilegia la chiarezza rispetto alle prestazioni, con commenti dettagliati e esempi pratici.

## 📋 Caratteristiche

- **Strutture dati matematiche**: Vector e Matrix con operazioni ottimizzate
- **Funzioni di attivazione**: Sigmoid, Tanh, ReLU
- **Funzioni di loss**: Mean Squared Error (MSE), Cross Entropy
- **Inizializzazione pesi**: Random, Xavier, He
- **Training**: Stochastic Gradient Descent (SGD) con supporto mini-batch
- **Serializzazione**: Salvataggio e caricamento modelli in formato testuale
- **Validazione**: Calcolo accuracy e loss su test set
- **Monitoring**: TrainingMonitor per tracciare progresso del training
- **Error handling**: Gestione robusta degli errori con messaggi descrittivi

## 🛠️ Requisiti

- Compilatore C++17 (g++ 7.0+ o clang++ 5.0+)
- Make
- RapidCheck e GoogleTest (opzionali, solo per property test)

## 📦 Compilazione

### Build Standard
```bash
# Compila tutti gli esempi (configurazione default)
make

# Compila con simboli di debug
make debug

# Compila con ottimizzazioni
make release
```

### Build Libreria

Il framework può essere compilato come libreria condivisa (shared) e statica per essere riutilizzato in altri progetti.

```bash
# Compila sia shared che static library
make lib

# Questo crea:
# - lib/libneuralnet.so.1.0.0 (shared library)
# - lib/libneuralnet.so.1 (symlink)
# - lib/libneuralnet.so (symlink)
# - lib/libneuralnet.a (static library)
```

### Installazione Sistema

```bash
# Installa la libreria e gli header in /usr/local (richiede sudo)
sudo make install

# Oppure installa in una directory custom
make install PREFIX=$HOME/.local

# Disinstalla
sudo make uninstall
```

Dopo l'installazione, la libreria sarà disponibile per tutti i tuoi progetti:
- Libreria: `/usr/local/lib/libneuralnet.so*` e `/usr/local/lib/libneuralnet.a`
- Headers: `/usr/local/include/neuralnet/*.h`

### Build Specifici
```bash
# Compila solo gli esempi
make examples

# Compila solo i test
make tests

# Compila solo i property test (richiede RapidCheck)
make prop_tests
```

### Pulizia
```bash
# Rimuove tutti gli artifacts di build
make clean

# Pulisce e ricompila
make rebuild
```

## 🚀 Esempi

Il framework include diversi esempi educativi che dimostrano vari aspetti delle reti neurali.

### Eseguire gli Esempi

```bash
# Esegui tutti gli esempi in sequenza
make run_all

# Oppure esegui esempi individuali
make run_xor          # Problema XOR (non linearmente separabile)
make run_binary       # Classificazione binaria lineare
make run_save_load    # Serializzazione modelli
make run_mnist        # Riconoscimento cifre MNIST
make run_iris         # Classificazione Iris dataset
make run_sin          # Regressione funzione sin(x)
```

### 1. XOR Problem (`examples/xor_example.cpp`)

Il classico problema XOR dimostra che le reti neurali possono risolvere problemi non linearmente separabili.

**Caratteristiche:**
- Dataset: 4 esempi XOR `{[0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0}`
- Architettura: [2, 4, 1] con sigmoid
- Training: 10000 epochs, learning_rate=0.5
- Risultato atteso: >90% accuracy

```bash
make run_xor
```

### 2. Binary Classification (`examples/binary_classification.cpp`)

Classificazione binaria su dataset linearmente separabile generato sinteticamente.

**Caratteristiche:**
- Dataset: 200 punti 2D con decision boundary lineare
- Architettura: [2, 3, 1] con tanh
- Split: 80% train, 20% test
- Risultato atteso: 100% accuracy

```bash
make run_binary
```

### 3. Save/Load Example (`examples/save_load_example.cpp`)

Dimostra la serializzazione e deserializzazione dei modelli.

**Caratteristiche:**
- Addestra rete su XOR
- Salva modello in `xor_model.txt`
- Carica modello e verifica predizioni identiche
- Mostra formato di serializzazione

```bash
make run_save_load
```

### 4. MNIST Digit Recognition (`examples/mnist_example.cpp`)

Riconoscimento di cifre scritte a mano dal famoso dataset MNIST.

**Caratteristiche:**
- Dataset: 10000 immagini training, 10000 test (subset)
- Immagini: 28x28 pixel, normalizzate a [0,1]
- Architettura: [784, 128, 64, 10] con ReLU e sigmoid
- Training: 10 epochs, learning_rate=0.01
- Risultato atteso: >85% test accuracy
- Include visualizzazione ASCII delle predizioni

**Preparazione dataset:**
Il dataset MNIST deve essere nella cartella `mnist-dataset/`:
- `train-images.idx3-ubyte` (60000 immagini training)
- `train-labels.idx1-ubyte` (60000 labels training)
- `t10k-images.idx3-ubyte` (10000 immagini test)
- `t10k-labels.idx1-ubyte` (10000 labels test)

Puoi scaricare il dataset da: https://www.kaggle.com/

```bash
make run_mnist
```

### 5. Iris Classification (`examples/iris_example.cpp`)

Classificazione multi-classe sul classico dataset Iris di Ronald Fisher (1936).

**Caratteristiche:**
- Dataset: 150 campioni, 4 features (misure fiori), 3 classi
- Architettura: [4, 6, 3] con tanh e sigmoid
- Normalizzazione features a [0,1]
- Training: 300 epochs, learning_rate=0.15
- Risultato atteso: >90% test accuracy
- Dimostra classificazione su dati tabulari

```bash
make run_iris
```

### 6. Sin(x) Regression (`examples/sin_regression_example.cpp`)

Approssimazione della funzione sin(x) - dimostra regressione (output continuo).

**Caratteristiche:**
- Funzione: y = sin(x)
- Training range: [-π, π] con 100 campioni
- Test range: [-2π, 2π] (include extrapolation)
- Architettura: [1, 16, 16, 1] con tanh
- Training: 1000 epochs, learning_rate=0.01
- Include grafici ASCII della funzione appresa
- Dimostra interpolazione vs extrapolation

```bash
make run_sin
```

## 🧪 Testing

Il framework include due tipi di test:

### Unit Test
Test tradizionali che verificano comportamenti specifici:

```bash
# Esegui tutti i unit test
make test

# Oppure
make test_unit
```

### Property Test (RapidCheck)
Test basati su proprietà che verificano invarianti universali:

```bash
# Esegui tutti i property test
make test_prop

# Oppure
make test_property
```

### Tutti i Test
```bash
# Esegui unit test + property test
make test_all
```

**Property test disponibili:**
- Vector operations (commutatività, correttezza)
- Matrix operations (transpose involution, associatività)
- Activation functions (correttezza matematica, derivate)
- Loss functions (correttezza, gradienti)
- Network creation (topologia, connettività)
- Backpropagation (gradient checking)
- Training (convergenza, batch averaging)
- Serialization (round-trip, integrità)
- Error handling (validazione dimensioni, configurazioni)

## 🏗️ Architettura

### Componenti Principali

```
include/
├── vector.h           # Vettore matematico con operazioni
├── matrix.h           # Matrice con moltiplicazione e transpose
├── activation.h       # Funzioni di attivazione (Sigmoid, Tanh, ReLU)
├── loss.h             # Funzioni di loss (MSE, CrossEntropy)
├── layer.h            # Layer con forward pass e inizializzazione pesi
├── network.h          # Rete neurale con backpropagation e training
├── serializer.h       # Serializzazione/deserializzazione modelli
├── training_monitor.h # Monitoring progresso training
├── mnist_loader.h     # Utility per caricare dataset MNIST
└── validation.h       # Validazione e metriche

src/
├── vector.cpp
├── matrix.cpp
├── activation.cpp
├── loss.cpp
├── layer.cpp
├── network.cpp
├── serializer.cpp
└── training_monitor.cpp

examples/
├── xor_example.cpp
├── binary_classification.cpp
├── save_load_example.cpp
├── mnist_example.cpp
├── iris_example.cpp
└── sin_regression_example.cpp

tests/
├── test_*.cpp         # Unit test
└── prop_*.cpp         # Property test (RapidCheck)
```

### Flusso di Esecuzione

1. **Creazione Network**: `Network net; net.addLayer(input, hidden, activation);`
2. **Inizializzazione**: `layer.initializeXavier(fan_in, fan_out);`
3. **Training**: `net.train(inputs, targets, epochs, lr, loss_fn);`
4. **Predizione**: `Vector output = net.predict(input);`
5. **Validazione**: `double acc = net.calculateAccuracy(test_inputs, test_targets);`
6. **Serializzazione**: `net.save("model.txt");` / `net.load("model.txt");`

## 💻 Uso Base del Framework

### Uso come Libreria Installata

Se hai installato la libreria con `make install`, puoi usarla nei tuoi progetti:

```cpp
// my_project.cpp
#include <neuralnet/network.h>
#include <neuralnet/activation.h>
#include <neuralnet/loss.h>
#include <memory>
#include <iostream>

int main() {
    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();
    network.addLayer(2, 4, sigmoid);
    network.addLayer(4, 1, sigmoid);
    
    // ... training code ...
    
    return 0;
}
```

Compila con:
```bash
# Usando la shared library
g++ -std=c++17 my_project.cpp -lneuralnet -o my_project

# Usando la static library
g++ -std=c++17 my_project.cpp -static -lneuralnet -o my_project

# Se installata in path custom
g++ -std=c++17 -I$HOME/.local/include/neuralnet -L$HOME/.local/lib my_project.cpp -lneuralnet -o my_project
```

### Uso Diretto (Senza Installazione)

Se preferisci non installare la libreria, puoi linkare direttamente gli object files:

### Esempio Minimo

```cpp
#include "network.h"
#include "activation.h"
#include "loss.h"
#include <memory>

int main() {
    // Crea rete [2, 4, 1]
    Network network;
    auto sigmoid = std::make_shared<Sigmoid>();
    network.addLayer(2, 4, sigmoid);
    network.addLayer(4, 1, sigmoid);
    
    // Inizializza pesi
    std::srand(42);
    network.getLayer(0).initializeXavier(2, 4);
    network.getLayer(1).initializeXavier(4, 1);
    
    // Prepara dati XOR
    std::vector<Vector> inputs = {
        Vector{0, 0}, Vector{0, 1},
        Vector{1, 0}, Vector{1, 1}
    };
    std::vector<Vector> targets = {
        Vector{0}, Vector{1},
        Vector{1}, Vector{0}
    };
    
    // Addestra
    MeanSquaredError loss;
    network.train(inputs, targets, 10000, 0.5, loss);
    
    // Testa
    for (size_t i = 0; i < inputs.size(); ++i) {
        Vector pred = network.predict(inputs[i]);
        std::cout << "Input: [" << inputs[i][0] << ", " << inputs[i][1] 
                  << "] -> Output: " << pred[0] << std::endl;
    }
    
    return 0;
}
```

### Compilazione Esempio Custom (Senza Installazione)

```bash
# Compila prima gli object files del framework
make obj/activation.o obj/layer.o obj/loss.o obj/matrix.o obj/network.o obj/serializer.o obj/training_monitor.o obj/vector.o

# Poi compila il tuo esempio linkando gli object files
g++ -std=c++17 -Iinclude my_example.cpp obj/*.o -o my_example
./my_example
```

### Compilazione con Libreria Locale (Senza Installazione)

```bash
# Compila la libreria
make lib

# Compila il tuo progetto linkando la libreria locale
g++ -std=c++17 -Iinclude my_example.cpp -Llib -lneuralnet -o my_example

# Esegui specificando il path della libreria
LD_LIBRARY_PATH=./lib ./my_example
```

## 🔧 Eclipse CDT Integration

Il progetto è compatibile con Eclipse CDT per sviluppo e debugging.

### Import Progetto

1. File → Import → General → Existing Projects into Workspace
2. Seleziona la directory del progetto
3. Il Makefile verrà riconosciuto automaticamente

### Build in Eclipse

1. Project → Build Project (Ctrl+B)
2. Oppure usa i target specifici: Project → Build Configurations → Set Active

### Debug in Eclipse

1. Run → Debug Configurations
2. Crea nuova "C/C++ Application"
3. Seleziona l'eseguibile in `bin/`
4. Imposta breakpoint e avvia debug

### Configurazione Consigliata

- Build Command: `make`
- Clean Command: `make clean`
- Indexer: Usa "GCC Built-in Compiler Settings"
- Include Paths: aggiungi `${workspace_loc:/project_name/include}`

## 📚 Concetti Implementati

### 1. Forward Pass
Calcolo dell'output della rete dato un input:
```
z^l = W^l * a^(l-1) + b^l
a^l = activation(z^l)
```

### 2. Backpropagation
Calcolo dei gradienti per aggiornare i pesi:
```
δ^L = ∇loss ⊙ activation'(z^L)
δ^l = (W^(l+1))^T * δ^(l+1) ⊙ activation'(z^l)
∂L/∂W^l = δ^l * (a^(l-1))^T
∂L/∂b^l = δ^l
```

### 3. Gradient Descent
Aggiornamento parametri:
```
W_new = W_old - η * ∂L/∂W
b_new = b_old - η * ∂L/∂b
```

### 4. Inizializzazione Pesi

**Xavier (Glorot)**: Ottimale per sigmoid/tanh
```
W ~ U[-√(6/(fan_in + fan_out)), √(6/(fan_in + fan_out))]
```

**He**: Ottimale per ReLU
```
W ~ U[-√(2/fan_in), √(2/fan_in)]
```

### 5. Batch Training

- **SGD** (batch_size=1): Aggiorna dopo ogni esempio
- **Mini-batch** (batch_size=32): Aggiorna dopo 32 esempi
- **Batch** (batch_size=dataset_size): Aggiorna dopo tutto il dataset

## 🎓 Risorse Educative

### Documentazione Interna
- Ogni file sorgente include commenti dettagliati
- Gli esempi contengono spiegazioni passo-passo
- I test dimostrano l'uso corretto delle API

### Concetti Chiave da Studiare
1. **Universal Function Approximation**: Le reti neurali possono approssimare qualsiasi funzione continua
2. **Vanishing/Exploding Gradients**: Perché l'inizializzazione dei pesi è importante
3. **Overfitting**: Quando la rete memorizza invece di generalizzare (vedi Iris example)
4. **Interpolation vs Extrapolation**: Le reti funzionano meglio dentro il range di training (vedi sin example)
5. **Non-linear Separability**: Perché servono hidden layers (vedi XOR example)

### Tutorial Consigliati
- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) di Michael Nielsen
- [Deep Learning Book](https://www.deeplearningbook.org/) di Goodfellow, Bengio, Courville
- [CS231n](http://cs231n.stanford.edu/) - Stanford Course on CNNs

## 🐛 Troubleshooting

### Errori di Compilazione

**Errore**: `error: 'make_shared' is not a member of 'std'`
- Soluzione: Verifica di usare `-std=c++17` o superiore

**Errore**: `undefined reference to RapidCheck`
- Soluzione: Installa RapidCheck o compila solo gli esempi con `make examples`

### Errori di Runtime

**Errore**: `Dimension mismatch`
- Causa: Input size non corrisponde al primo layer
- Soluzione: Verifica che `input.size() == network.getLayer(0).inputSize()`

**Errore**: `NaN detected in forward pass`
- Causa: Pesi non inizializzati o learning rate troppo alto
- Soluzione: Chiama `initializeXavier()` o riduci learning rate

### Performance

**Training troppo lento**:
- Usa `make release` per compilare con ottimizzazioni (-O3)
- Riduci il numero di epochs o usa un subset del dataset
- Aumenta il batch_size per ridurre gli aggiornamenti

**Accuracy bassa**:
- Aumenta epochs o hidden neurons
- Prova diverse funzioni di attivazione
- Verifica che i dati siano normalizzati
- Controlla che learning rate non sia troppo alto/basso

### Riconoscere e Prevenire l'Overfitting

L'**overfitting** si verifica quando la rete memorizza i dati di training invece di imparare pattern generalizzabili. È uno dei problemi più comuni nel machine learning.

#### 🔍 Come Riconoscere l'Overfitting

**Segnali principali:**

1. **Gap Train-Test Accuracy**
   ```
   Epoch 100: Train Acc = 98%, Test Acc = 75%  ⚠️ OVERFITTING!
   Epoch 100: Train Acc = 92%, Test Acc = 90%  ✓ Buona generalizzazione
   ```
   - Gap > 10-15%: probabile overfitting
   - Gap < 5%: buona generalizzazione

2. **Loss che Diverge**
   ```
   Epoch | Train Loss | Test Loss
   ------|------------|----------
   10    | 0.150      | 0.160     ✓
   50    | 0.050      | 0.080     ✓
   100   | 0.010      | 0.150     ⚠️ Test loss aumenta!
   ```
   - Train loss continua a scendere
   - Test loss smette di scendere o aumenta

3. **Accuracy che Peggiora**
   ```
   Epoch | Train Acc | Test Acc
   ------|-----------|----------
   50    | 85%       | 82%       ✓
   100   | 95%       | 80%       ⚠️ Test acc peggiora
   150   | 99%       | 78%       ⚠️ Overfitting grave
   ```

**Esempio pratico (vedi `iris_example.cpp`):**
```
# Con 8 neuroni nascosti (troppi per Iris):
Train Acc: 98.33%, Test Acc: 86.67%  → Gap 12% (overfitting)

# Con 6 neuroni nascosti (bilanciato):
Train Acc: 97.50%, Test Acc: 100%    → Gap 2.5% (ottimo!)
```

#### 🛡️ Strategie per Prevenire l'Overfitting

**1. Ridurre la Complessità del Modello**
```cpp
// Troppo complesso per dataset piccolo (150 samples)
network.addLayer(4, 16, tanh);  // ❌ Troppi parametri
network.addLayer(16, 8, tanh);
network.addLayer(8, 3, sigmoid);

// Appropriato per dataset piccolo
network.addLayer(4, 6, tanh);   // ✓ Meno parametri
network.addLayer(6, 3, sigmoid);
```

**Regola pratica:** Numero parametri ≈ 10% del numero di esempi training
- Iris: 120 esempi → ~12 parametri → architettura [4,6,3] ha 45 parametri (ok)
- MNIST: 10000 esempi → ~1000 parametri → architettura [784,128,64,10] ha ~110k (ok con più dati)

**2. Early Stopping**
```cpp
// Ferma il training quando test accuracy smette di migliorare
double best_test_acc = 0.0;
int patience = 0;
const int max_patience = 10;

for (size_t epoch = 0; epoch < max_epochs; ++epoch) {
    network.train(train_inputs, train_targets, 1, lr, loss);
    double test_acc = network.calculateAccuracy(test_inputs, test_targets);
    
    if (test_acc > best_test_acc) {
        best_test_acc = test_acc;
        patience = 0;
        // Salva il miglior modello
        network.save("best_model.txt");
    } else {
        patience++;
        if (patience >= max_patience) {
            std::cout << "Early stopping at epoch " << epoch << std::endl;
            break;
        }
    }
}
```

**3. Aumentare i Dati di Training**
```cpp
// Dataset piccolo → overfitting facile
std::vector<Vector> train_data(100);  // ❌ Troppo pochi

// Dataset più grande → generalizza meglio
std::vector<Vector> train_data(1000); // ✓ Meglio

// Data augmentation (per immagini)
// - Rotazioni, flip, crop, noise
// - Aumenta artificialmente il dataset
```

**4. Normalizzazione dei Dati**
```cpp
// Dati non normalizzati → overfitting più probabile
Vector input{150.0, 3.5, 45.0, 12.0};  // ❌ Scale diverse

// Dati normalizzati [0,1] → training più stabile
Vector input{0.75, 0.35, 0.45, 0.12};  // ✓ Normalizzati
```

**5. Regolarizzazione (Non implementata in questo framework)**
- **L2 Regularization**: Penalizza pesi grandi
- **Dropout**: Disattiva neuroni random durante training
- **Batch Normalization**: Normalizza attivazioni tra layer

**6. Validazione Incrociata (Cross-Validation)**
```cpp
// Invece di un singolo train/test split
// Usa K-fold cross-validation per valutazione più robusta

// Esempio: 5-fold CV
const int K = 5;
std::vector<double> fold_accuracies;

for (int fold = 0; fold < K; ++fold) {
    // Dividi dataset in K parti
    // Usa fold i come test, resto come training
    // Addestra e valuta
    fold_accuracies.push_back(test_accuracy);
}

double avg_accuracy = mean(fold_accuracies);
// Se std_dev è alta → modello instabile
```

#### 📊 Monitoraggio Durante il Training

**Cosa stampare per rilevare overfitting:**
```cpp
std::cout << "Epoch | Train Loss | Train Acc | Test Acc  | Gap" << std::endl;
std::cout << "------|------------|-----------|-----------|-----" << std::endl;

for (size_t epoch = 0; epoch < epochs; ++epoch) {
    network.train(train_inputs, train_targets, 1, lr, loss);
    
    double train_loss = network.validate(train_inputs, train_targets, loss);
    double train_acc = network.calculateAccuracy(train_inputs, train_targets);
    double test_acc = network.calculateAccuracy(test_inputs, test_targets);
    double gap = train_acc - test_acc;
    
    std::cout << std::setw(5) << epoch << " | "
              << std::fixed << std::setprecision(4) << train_loss << " | "
              << std::setprecision(2) << (train_acc * 100) << "%   | "
              << (test_acc * 100) << "%    | "
              << (gap * 100) << "%" << std::endl;
    
    // Allarme se gap > 15%
    if (gap > 0.15) {
        std::cout << "⚠️  Warning: Large train-test gap detected!" << std::endl;
    }
}
```

#### 🎯 Esempi nel Framework

**XOR (`xor_example.cpp`)**: 
- Dataset minuscolo (4 esempi) → overfitting inevitabile
- Soluzione: architettura molto semplice [2,4,1]

**Iris (`iris_example.cpp`)**:
- Dataset piccolo (120 train) → overfitting probabile
- Soluzione: ridotto neuroni da 8 a 6, epochs da 500 a 300
- Risultato: gap ridotto da 12% a 2.5%

**MNIST (`mnist_example.cpp`)**:
- Dataset grande (10000 train) → overfitting meno probabile
- Architettura più complessa [784,128,64,10] è ok
- Gap train-test sempre < 1.5% (ottimo!)

**Sin(x) (`sin_regression_example.cpp`)**:
- Dimostra interpolation vs extrapolation
- Dentro training range: MAE 0.034 (ottimo)
- Fuori training range: MAE 0.222 (peggiore)
- Mostra che le reti non extrapolano bene

#### 💡 Regole Pratiche

1. **Inizia semplice**: Pochi neuroni, poche epochs
2. **Monitora sempre**: Stampa train e test accuracy
3. **Guarda il gap**: Se > 10%, riduci complessità
4. **Usa validation set**: Non toccare mai il test set durante sviluppo
5. **Salva il migliore**: Non l'ultimo modello, ma quello con miglior test accuracy

**Ricorda**: Un modello che generalizza bene è meglio di uno che memorizza perfettamente!

## 📄 Licenza

Questo progetto è rilasciato per scopi educativi. Sentiti libero di usarlo, modificarlo e imparare da esso.

## 🤝 Contributi

Questo è un progetto educativo. Suggerimenti e miglioramenti sono benvenuti!

## 📞 Supporto

Per domande o problemi:
1. Leggi i commenti nel codice sorgente
2. Esegui gli esempi per vedere l'uso corretto
3. Consulta i test per capire il comportamento atteso

---

**Buon apprendimento! 🚀**
