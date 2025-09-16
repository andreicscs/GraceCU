#ifndef NN_DEBUG_H
#define NN_DEBUG_H

#ifdef __cplusplus
extern "C" {
#endif

// Enable CRT memory leak detection (Windows only)
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <string.h>
#include <stdbool.h>

// include the public API
#include "NeuralNetwork.h"
#include "Matrix.h"

// if the debug header is being used, don't define the struct twice
#ifndef NN_STRUCT
#define NN_STRUCT
struct NeuralNetwork {
    const unsigned int* architecture;
    unsigned int layerCount;
    Matrix* weights;
    Matrix* biases;
    Matrix* weightsGradients;
    Matrix* biasesGradients;
    Matrix* outputs;
    Matrix* activations;
    unsigned int numOutputs;
    NNConfig config;
};
#endif

// Internal forward/backward computations
bool forward(NeuralNetwork nn, Matrix input);
bool backPropagation(NeuralNetwork nn, Matrix expectedOutput, Matrix input);
bool updateWeightsAndBiases(NeuralNetwork nn, unsigned int batchSize);

// Internal activation functions
Matrix applyActivation(const NeuralNetwork nn, Matrix matrix, unsigned int iLayer);
Matrix multipleOutputActivationFunction(Matrix mat, int af);
float activationFunction(float x, int af);
float AFDerivative(float x, int af);
float relu(float x);
float reluDerivative(float x);
float sigmoid(float x);
float sigmoidDerivative(float sig);
Matrix softmax(Matrix mat);

// Internal loss functions
float loss(float output, float expectedOutput, int lf);
float lossDerivative(float output, float expectedOutput, int lf);
float multipleOutputLoss(Matrix output, Matrix expectedOutput, int lf);
float CCEloss(Matrix predictions, Matrix labels);
Matrix CCElossDerivativeMatrix(Matrix predicted, Matrix expected);
float BCEloss(float output, float expectedOutput);
float BCElossDerivative(float output, float expectedOutput);
float MSEloss(float output, float expectedOutput);
float MSElossDerivative(float output, float expectedOutput);

// Backprop helpers
Matrix computeOutputLayerPartialGradients(const NeuralNetwork nn, Matrix expectedOutput);
Matrix computeActivationDerivative(Matrix outputs, int af);
Matrix computeMultipleOutputLossDerivativeMatrix(Matrix output, Matrix expectedOutput, int lf);
Matrix computeLossDerivative(Matrix outputs, Matrix expectedOutputs, int lf);

// Initialization and random helpers
void initializeMatrixRand(Matrix mat, float mean, float stddev);
float randomNormal(float mean, float stddev);
float initializationFunction(NNInitializationFunction weightInitializerF, unsigned int nIn, unsigned int nOut);

// Validation helpers
NNStatus checkNNConfig(NNConfig config);
NNStatus allocateNNMatricesArrays(NeuralNetwork* nn);
NNStatus validateNNSettings(const NeuralNetwork* nn);
const char* NNStatusToString(NNStatus code);

// Nn testing helpers
bool computeSingleOutputAccuracy(const NeuralNetwork nn, Matrix dataset, float* out);
bool computeMultiClassAccuracy(const NeuralNetwork nn, Matrix dataset, float* out);

#ifdef __cplusplus
} // end extern "C"
#endif
#endif // NN_DEBUG_H
