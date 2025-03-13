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

/**
* This function creates, allocates memory, and initializes a neuralNetwork structure.
*
* @param *architecture: pointer to the architecture array, each value rapresents the number of neurons of that layer. EX. The first layer (architecture[0]) is the input layer, the last layer is the output layer.
* @param layerCount: the length of the architecture array, i.e. the number of layers of the neural network.
* @param config: structure that contains options for the neuralNetwork that can be set by the user.
*
* @return NeuralNetwork*: returns a pointer to the created nn.
*/
NeuralNetwork* createNeuralNetwork(const int *architecture, const unsigned int layerCount, NNConfig config);

/**
* This function frees the allocated memory of a neuralNetwork.
*
* @param *nn: pointer to the neuralNetwork data structure.
*
*/
void freeNeuralNetwork(NeuralNetwork *nn);

/**
* This function trains the neural network on a dataset.
*
* @param *nn: pointer to the neuralNetwork data structure.
* @param trainingData: all columns dedicated to input apart from the last *nOutputs columns which will be used to store the expected output
* @param batchSize: the size of the single batches the training data will be split in, after which the weights and biases update. 1 for Stochastic gradient descent, 1<batchSize<trainingDataSize for mini batches, trainingDataSize for full batch training
*
*/
void trainNN(NeuralNetwork *nn, Matrix trainingData, unsigned int batchSize);

/**
* This function saves the current state of the neural network allowing it to be used without training.
*
* @param *nn: pointer to the neuralNetwork data structure.
*
*/
void saveStateNN(NeuralNetwork *nn);

/**
* This function loads the saved state of the neural network from a file.
*
* @param *filename: path of the file where the saved nn is stored.
*
*/
NeuralNetwork* loadStateNN(const char* filename);

/**
* This function uses the already trained neuralNetwork to predict the output of a given input.
*
* @param *nn: pointer to the neuralNetwork data structure.
* @param input: Matrix containing the input for the nn.
*
*/
Matrix predictNN(NeuralNetwork* nn, Matrix input);


/**
* This function computes the accuracy of the output of the neural network on a given dataset.
*
* @param *nn: pointer to the neuralNetwork data structure.
* @param dataset: all columns dedicated to input apart from the last *nOutputs columns which will be used to store the expected output
*
* @return float: accuracy of the nn
*/
float computeAccuracyNN(NeuralNetwork *nn, Matrix dataset);

/**
* This function computes the average loss of the output of the neural network on a given dataset.
*
* @param *nn: pointer to the neuralNetwork data structure.
* @param trainingData: all columns dedicated to input apart from the last *nOutputs columns which will be used to store the expected output
* 
* @return float: average loss of the nn
*/
float computeAverageLossNN(NeuralNetwork *nn, Matrix trainingData);

#endif // NN_H