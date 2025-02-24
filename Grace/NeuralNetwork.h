#ifndef NN_H
#define NN_H

#include "Matrix.h"

typedef struct {
    int* architecture;
    int layerCount;
    Matrix* weights;
    Matrix* biases;
    Matrix* weightsGradients;
    Matrix* biasesGradients;
    Matrix* inputGradients;
    Matrix* outputs;
    Matrix* activations;
    double learningRate;
    char* hiddenLayersAF;
    char* outputLayerAF;
    char* lossFunction;
    int numOutputs;
} NeuralNetwork;

NeuralNetwork createNeuralNetwork(int* architecture, int layerCount);
void freeNeuralNetwork(NeuralNetwork nn);

void train(Matrix trainingData, int nOutputs, int batchSize);
void saveState(NeuralNetwork nn);
double computeAccuracy(NeuralNetwork nn);
double computeAverageLoss(NeuralNetwork nn);


#endif // NN_H