#ifndef NN_H
#define NN_H

#include "Matrix.h"

#define NN_SIGMOID 0
#define NN_RELU 1
#define NN_SOFTMAX 10
#define NN_CCE 20
#define NN_MSE 21
#define NN_BCE 22
#define NN_epsilon 1e-10f // small value
#define NN_invalidP nullptr

struct NeuralNetwork;
struct NNConfig {
    float learningRate;
    int hiddenLayersAF;
    int outputLayerAF;
    int lossFunction;
};

NeuralNetwork* createNeuralNetwork(int *architecture, int layerCount, NNConfig config);
void freeNeuralNetwork(NeuralNetwork *nn);

void trainNN(NeuralNetwork *nn, Matrix trainingData, int batchSize);
void saveStateNN(NeuralNetwork *nn);
float computeAccuracyNN(NeuralNetwork *nn, Matrix dataset);
float computeAverageLossNN(NeuralNetwork *nn, Matrix trainingData);

#endif // NN_H