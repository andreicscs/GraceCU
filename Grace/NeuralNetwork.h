#ifndef NN_H
#define NN_H

/**
* 
* A simple NeuralNetwork library that implements basic functions to be able to train and test a neural network.
* 
*/

#include "Matrix.h"
typedef enum {
    NN_OK,                      // no errors
    NN_ERROR_INVALID_ARGUMENT,   // invalid input arguments
    NN_ERROR_MEMORY_ALLOCATION, // memory allocation failed
    NN_ERROR_INVALID_CONFIG,     // invalid neural network configuration
    NN_ERROR_INVALID_LEARNING_RATE, // learning rate is invalid
    NN_ERROR_UNSUPPORTED_HIDDEN_AF, // unsupported hidden layer activation function
    NN_ERROR_UNSUPPORTED_OUTPUT_AF, // unsupported output layer activation function
    NN_ERROR_INVALID_LOSS_FUNCTION, // invalid loss function
    NN_ERROR_INVALID_LOSS_AF_COMBINATION, // invalid combination of loss function and activation
    NN_ERROR_IO_FILE, // error while storing / loading NeuralNetwork struct.
    NN_ERROR_UNKNOWN,   // error was not recognized
} NNStatus;


typedef enum {
    NN_INITIALIZATION_0,
    NN_INITIALIZATION_1,
    NN_INITIALIZATION_SPARSE,
    NN_INITIALIZATION_HE,
    NN_INITIALIZATION_HE_UNIFORM,
    NN_INITIALIZATION_XAVIER,
    NN_INITIALIZATION_XAVIER_UNIFORM,

} NNInitializationFunction;
typedef enum {
    NN_ACTIVATION_SIGMOID,
    NN_ACTIVATION_RELU,
    NN_ACTIVATION_SOFTMAX,
} NNActivationFunction;
typedef enum {
    NN_LOSS_CCE,
    NN_LOSS_MSE,
    NN_LOSS_BCE,
} NNLossFunction;

#define NN_epsilon 1e-10f // small value
#define NN_invalidP NULL

typedef struct NeuralNetwork NeuralNetwork;

typedef struct NNConfig {
    float learningRate;
    NNInitializationFunction weightInitializerF;
    NNActivationFunction hiddenLayersAF;
    NNActivationFunction outputLayerAF;
    NNLossFunction lossFunction;
}NNConfig;

#ifdef __cplusplus // for c++ compatibility
extern "C" {
#endif

/**
* This function creates, allocates memory, and initializes a neuralNetwork structure.
*
* @param *architecture: pointer to the architecture array, each value rapresents the number of neurons of that layer. EX. The first layer (architecture[0]) is the input layer, the last layer is the output layer.
* @param layerCount: the length of the architecture array, i.e. the number of layers of the neural network.
* @param config: structure that contains options for the neuralNetwork that can be set by the user.
* @param **nnP: pointer to the pointer of the neuralNetwork data structure, the function will use this address to return the nn.
*
* @return NNStatus: returns error code.
*/
NNStatus createNeuralNetwork(const unsigned int *architecture, const unsigned int layerCount, NNConfig config, NeuralNetwork **nnP);

/**
* This function frees the allocated memory of a neuralNetwork.
*
* @param *nn: pointer to the neuralNetwork data structure.
*
* @return NNStatus: returns error code.
*/
NNStatus freeNeuralNetwork(NeuralNetwork *nn);

/**
* This function trains the neural network on a dataset.
*
* @param *nn: pointer to the neuralNetwork data structure.
* @param trainingData: all columns dedicated to input apart from the last *nOutputs columns which will be used to store the expected output
* @param batchSize: the size of the single batches the training data will be split in, after which the weights and biases update. 1 for Stochastic gradient descent, 1<batchSize<trainingDataSize for mini batches, trainingDataSize for full batch training
*
* @return NNStatus: returns error code.
*/
NNStatus trainNN(NeuralNetwork *nn, Matrix trainingData, unsigned int batchSize);

/**
* This function saves the current state of the neural network allowing it to be used without training. doesn't close file!
* all values are saved allowing to continue training after reloading the nn.
*
* @param fpOut: the file the nn will be stored in
* @param *nn: pointer to the neuralNetwork data structure.
*
* @return NNStatus: returns error code.
*/
NNStatus saveStateNN(NeuralNetwork *nn, FILE *fpOut);

/**
* This function loads the saved state of the neural network from a file. doesn't close file!
* 
* @param fpIn: the file where the nn is stored
* @param **nnP: pointer to the pointer of the neuralNetwork data structure, the function will use this address to return the nn.
*
* @return NNStatus: returns error code.
*/
NNStatus loadStateNN(FILE *fpIn, NeuralNetwork **nnP);

/**
* This function uses the already trained neuralNetwork to predict the output of a given input.
*
* 
* @param *nn: pointer to the neuralNetwork data structure.
* @param input: Matrix containing the input for the nn.
* @param *output: Matrix address where the activations of the last layer of the nn will be returned.
*
* @return NNStatus: returns error code.
* 
*/
NNStatus predictNN(NeuralNetwork *nn, Matrix input, Matrix *output);


/**
* This function computes the accuracy of the output of the neural network on a given dataset.
*
* 
* @param *nn: pointer to the neuralNetwork data structure.
* @param dataset: all columns dedicated to input apart from the last *nOutputs columns which will be used to store the expected output
* @param *acuracy: the float address where the accuracy of the nn wil be returned
*
* @return NNStatus: returns error code.
*
*/
NNStatus computeAccuracyNN(NeuralNetwork *nn, Matrix dataset, float *accuracy);

/**
* This function computes the average loss of the output of the neural network on a given dataset.
*
* @param *nn: pointer to the neuralNetwork data structure.
* @param trainingData: all columns dedicated to input apart from the last *nOutputs columns which will be used to store the expected output
* @param *averageLoss: the float address where the averageLoss of the nn wil be returned
* 
* @return NNStatus: returns error code.
*/
NNStatus computeAverageLossNN(NeuralNetwork *nn, Matrix trainingData, float *averageLoss);


/**
* This function converts NNStatus codes to string for a clearer understanding of what the error code means.
*
* @param code: status code returned from the library.
*
* @return const char*: returns status code message.
*/
const char* NNStatusToString(NNStatus code);


#ifdef __cplusplus
} // end extern "C"
#endif

#endif // NN_H