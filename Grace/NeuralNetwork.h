#ifndef NN_H
#define NN_H

#include "Matrix.h"



#define NN_SIGMOID 0
#define NN_RELU 1
#define NN_SOFTMAX 10
#define NN_CCE 20
#define NN_MSE 21
#define NN_BCE 22


/*
*
*   !! TO DO implement a different struct for NN config options to improve information hiding
*   and implement a better and more flexible library.
*
*/

typedef struct {
    int* architecture;
    int layerCount;
    Matrix* weights;
    Matrix* biases;
    Matrix* weightsGradients;
    Matrix* biasesGradients;
    Matrix* outputs;
    Matrix* activations;
    Matrix* deltas;
    double learningRate;
    int hiddenLayersAF;
    int outputLayerAF;
    int lossFunction;
    int numOutputs;
} NeuralNetwork;

NeuralNetwork createNeuralNetwork(int* architecture, int layerCount);
void freeNeuralNetwork(NeuralNetwork nn);

void trainNN(NeuralNetwork nn, Matrix trainingData , int batchSize);
void saveStateNN(NeuralNetwork nn);
double computeAccuracyNN(NeuralNetwork nn, Matrix dataset);
double computeAverageLossNN(NeuralNetwork nn, Matrix trainingData);


#endif // NN_H