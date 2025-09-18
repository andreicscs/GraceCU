# 📘 NeuralNetwork.h — API Reference

GraceCU provides a lightweight C/C++ neural network library with a focus on clarity and educational value.  
This document describes the public API exposed in `NeuralNetwork.h`.

---

## ⚙️ Initialization

### `void createNNConfig(NNConfig *config)`
Initializes a `NNConfig` struct with **invalid values**, ensuring unset parameters can later be detected and validated.  
- **config** → pointer to the config struct to initialize  

**Postcondition:**  
- The config is safe to check for missing/invalid values before use.  

---

## 🔧 Configuration Structures

### `NNConfig`
Holds all user-defined hyperparameters:  
```c
typedef struct NNConfig {
    float learningRate;
    NNInitializationFunction weightInitializerF;
    NNActivationFunction hiddenLayersAF;
    NNActivationFunction outputLayerAF;
    NNLossFunction lossFunction;
} NNConfig;
```

---

## 🏗️ Network Creation & Destruction

### `NNStatus createNeuralNetwork(const unsigned int *architecture, const unsigned int layerCount, NNConfig config, NeuralNetwork **nnP)`
Creates, allocates, and initializes a new neural network.  

- **architecture** → array of neuron counts per layer (e.g. `{784, 128, 10}`)  
- **layerCount** → length of the architecture array  
- **config** → hyperparameters (learning rate, activation functions, weight init, etc.)  
- **nnP** → output pointer; on success points to a valid `NeuralNetwork` object  

**Preconditions:**  
- `config` must be initialized via `createNNConfig()`  
- `layerCount` must match the length of `architecture`  
- User must initialize RNG seed (`srand((unsigned int)time(NULL));`)  

**Postconditions:**  
- `*nnP` points to a heap-allocated network  
- Weights and biases initialized according to `config`  

---

### `NNStatus freeNeuralNetwork(NeuralNetwork *nn)`
Frees all memory associated with a neural network.  

- **nn** → pointer to a previously created network  

**Precondition:**  
- `nn` must have been created via `createNeuralNetwork()`  

**Postcondition:**  
- All memory allocated for the network is released  

---

## 🎓 Training & Evaluation

### `NNStatus trainNN(NeuralNetwork *nn, const Matrix trainingData, const unsigned int batchSize)`
Trains the neural network using backpropagation and gradient descent.  

- **trainingData** → matrix with input columns + last `nOutputs` columns as expected outputs  
- **batchSize** → training mode depends on this value:  
  - `1` → stochastic gradient descent  
  - `1 < batchSize < trainingSetSize` → mini-batch training  
  - `batchSize == trainingSetSize` → full batch training  

**Postcondition:**  
- Weights and biases updated according to gradients  

---

### `NNStatus computeAccuracyNN(const NeuralNetwork *nn, Matrix dataset, float *accuracy)`
Computes classification accuracy of the network on a dataset.  

- **dataset** → matrix formatted like training data  
- **accuracy** → pointer to float where result (0–100%) is stored  

---

### `NNStatus computeAverageLossNN(const NeuralNetwork *nn, Matrix trainingData, float *averageLoss)`
Computes average loss of the network on a dataset.  

- **trainingData** → formatted dataset  
- **averageLoss** → pointer to float where average loss is stored  

---

## 🔮 Prediction

### `NNStatus predictNN(const NeuralNetwork *nn, Matrix input, Matrix *output)`
Runs a forward pass on the input data.  

- **input** → matrix containing input samples  
- **output** → pointer to matrix where activations of the last layer will be stored  

---

## 💾 Persistence

### `NNStatus saveStateNN(const NeuralNetwork *nn, FILE *fpOut)`
Saves the **entire network state** (weights, biases, gradients, config, etc.) to a file.  

- **fpOut** → file pointer (must already be opened for writing)  
- **nn** → pointer to the network  

⚠️ Does **not** close the file.  

---

### `NNStatus loadStateNN(FILE *fpIn, NeuralNetwork **nnP)`
Loads a previously saved network state from a file.  

- **fpIn** → file pointer (must already be opened for reading)  
- **nnP** → output pointer where the loaded network will be stored  

⚠️ Does **not** close the file.  

---

## ❌ Error Handling

### `const char* NNStatusToString(NNStatus code)`
Converts an error/status code into a human-readable string.  

- **code** → status code returned by API functions  
- **returns** → string message (e.g. `"NN_ERROR_INVALID_ARGUMENT"`)
