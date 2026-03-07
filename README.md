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

Puoi scaricare il dataset da: http://yann.lecun.com/exdb/mnist/

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

### 6. Sin(x) Regression (`examples/sin_regression.cpp`)

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
├── mnist_loader.h     # Utility per caricare dataset MNIST
├── iris_example.cpp
└── sin_regression.cpp

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

### Compilazione Esempio Custom

```bash
g++ -std=c++17 -Iinclude my_example.cpp obj/*.o -o my_example
./my_example
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

