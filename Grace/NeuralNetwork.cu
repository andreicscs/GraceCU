// check memory leaks
#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>

#ifdef _DEBUG
#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new DEBUG_NEW
#define malloc(s) _malloc_dbg(s, _NORMAL_BLOCK, __FILE__, __LINE__)
#endif

#include <stdlib.h>
#include <time.h>
#include "NeuralNetwork.h"
#include "Matrix.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <iostream>
#include <stdio.h>


using namespace std;


struct NeuralNetwork {
    const int* architecture;
    unsigned int layerCount;
    Matrix* weights;
    Matrix* biases;
    Matrix* weightsGradients;
    Matrix* biasesGradients;
    Matrix* outputs;
    Matrix* activations;
    Matrix* deltas;
    unsigned int numOutputs;
    NNConfig config;
};

void forward(NeuralNetwork nn, Matrix input);
void backPropagation(NeuralNetwork nn, Matrix expectedOutput);
Matrix applyActivation(NeuralNetwork nn, Matrix matrix, int iLayer);
Matrix multipleOutputActivationFunction(Matrix mat, int af);
float activationFunction(float x, int af);
Matrix computeOutputLayerDeltas(NeuralNetwork nn, Matrix expectedOutput);
Matrix computeActivationDerivative(Matrix outputs, int af);
Matrix computeMultipleOutputLossDerivativeMatrix(Matrix output, Matrix expectedOutput, int lf);
float AFDerivative(float x, int af);
Matrix CCElossDerivativeMatrix(Matrix predicted, Matrix expected);
Matrix computeActivationDerivative(Matrix outputs, int af);
Matrix computeLossDerivative(Matrix outputs, Matrix expectedOutputs, int lf);
float lossDerivative(float output, float expectedOutput, int lf);
float BCEloss(float output, float expectedOutput);
float BCElossDerivative(float output, float expectedOutput);
float MSEloss(float output, float expectedOutput);
float MSElossDerivative(float output, float expectedOutput);
void initializeMatrixRand(Matrix mat, float mean, float stddev);
Matrix softmax(Matrix mat);
float reluDerivative(float x);
float relu(float x);
float sigmoid(float x);
void updateWeightsAndBiases(NeuralNetwork nn, int batchSize);
float loss(float output, float expectedOutput, int lf);
float multipleOutputLoss(Matrix output, Matrix expectedOutput, int lf);
float CCEloss(Matrix predictions, Matrix labels);
float computeSingleOutputAccuracy(NeuralNetwork nn, Matrix dataset);
float computeMultiClassAccuracy(NeuralNetwork nn, Matrix dataset);
float randomNormal(float mean, float stddev);
float sigmoidDerivative(float sig);


NeuralNetwork* createNeuralNetwork(const int *architecture, const unsigned int layerCount, NNConfig config) {
    if (architecture == NN_invalidP) throw "createNeuralNetwork: invalid arguments";

    NeuralNetwork* nn = (NeuralNetwork*) malloc(sizeof(NeuralNetwork));
    if (nn == NN_invalidP) throw "createNeuralNetwork: malloc failed for NeuralNetwork";
    
    // !! TO DO implement a checkConfig(NNConfig config) function to make sure the input config is allowed.
    nn->config = config;

    nn->architecture = architecture;
    nn->layerCount = layerCount;
    nn->weights = (Matrix*) malloc(sizeof(Matrix) * (layerCount-1)); if (nn->weights == NN_invalidP)throw "createNeuralNetwork: malloc failed";
    nn->biases = (Matrix*)malloc(sizeof(Matrix) * (layerCount-1)); if (nn->biases == NN_invalidP)throw "createNeuralNetwork: malloc failed";
    nn->weightsGradients = (Matrix*)malloc(sizeof(Matrix) * (layerCount-1)); if (nn->weightsGradients == NN_invalidP)throw "createNeuralNetwork: malloc failed";
    nn->biasesGradients = (Matrix*)malloc(sizeof(Matrix) * (layerCount-1)); if (nn->biasesGradients == NN_invalidP)throw "createNeuralNetwork: malloc failed";
    nn->outputs = (Matrix*)malloc(sizeof(Matrix) * layerCount); if (nn->outputs == NN_invalidP)throw "createNeuralNetwork: malloc failed";
    nn->activations = (Matrix*)malloc(sizeof(Matrix) * layerCount); if (nn->activations == NN_invalidP)throw "createNeuralNetwork: malloc failed";
    nn->deltas = (Matrix*)malloc(sizeof(Matrix) * (layerCount-1)); if (nn->deltas == NN_invalidP)throw "createNeuralNetwork: malloc failed";
    nn->numOutputs = architecture[layerCount - 1];


    srand((unsigned int)time(NULL));


    /*
     * Allocating and initializing weights, biases, and gradients for layers 1 to layerCount-1.
     * 
     *  !!TO DO
     * should implement different initialization methods for different activation functions, or maybe let the user decide which initialization method to use, 
     * initializing weights using he initialization and biases to 0.01(which helps prevent dead neurons) for now.
     */
     

    for (int i = 0; i < layerCount - 1; ++i) {
        nn->weights[i] = createMatrix(architecture[i], architecture[i + 1]);
        nn->biases[i] = createMatrix(1, architecture[i + 1]);
        nn->weightsGradients[i] = createMatrix(architecture[i], architecture[i + 1]);
        nn->biasesGradients[i] = createMatrix(1, architecture[i + 1]);
        nn->deltas[i] = createMatrix(1, architecture[i + 1]);

        // he initialization
        float stddev = sqrt(2.0f / nn->weights[i].rows);
        float mean = 0.0f;
        initializeMatrixRand(nn->weights[i], mean, stddev);
        fillMatrix(nn->biases[i], 0.01f);

        // Initialize gradients to zero
        fillMatrix(nn->weightsGradients[i], 0);
        fillMatrix(nn->biasesGradients[i], 0);
        fillMatrix(nn->deltas[i], 0);
        float sum = 0.0;
        for (int j = 0; j < nn->weights[i].rows * nn->weights[i].cols; ++j) {
            sum += fabs(nn->weights[i].elements[j]);
        }
    }

    for (int i = 0; i < layerCount; ++i) {
        nn->outputs[i] = { 0, 0, NN_invalidP };
        nn->activations[i] = { 0, 0, NN_invalidP };
    }

	return nn;
}

void freeNeuralNetwork(NeuralNetwork *nn) {
    if (nn == NN_invalidP) throw "freeNeuralNetwork: invalid argument";
    // free all matrices inside matrices arrays
    for (int i = 0; i < nn->layerCount; ++i) {
        if (nn->outputs[i].elements != NN_invalidP) freeMatrix(nn->outputs[i]);
        if (nn->activations[i].elements != NN_invalidP) freeMatrix(nn->activations[i]);
    }
    for (int i = 0; i < nn->layerCount - 1; ++i) {
        if (nn->weights[i].elements != NN_invalidP) freeMatrix(nn->weights[i]);
        if (nn->biases[i].elements != NN_invalidP) freeMatrix(nn->biases[i]);
        if (nn->weightsGradients[i].elements != NN_invalidP) freeMatrix(nn->weightsGradients[i]);
        if (nn->biasesGradients[i].elements != NN_invalidP) freeMatrix(nn->biasesGradients[i]);
        if (nn->deltas[i].elements != NN_invalidP) freeMatrix(nn->deltas[i]);
    }

    // free matrices arrays
    if (nn->weights != NN_invalidP) free(nn->weights);
    if (nn->biases != NN_invalidP) free(nn->biases);
    if (nn->weightsGradients != NN_invalidP) free(nn->weightsGradients);
    if (nn->biasesGradients != NN_invalidP) free(nn->biasesGradients);
    if (nn->outputs != NN_invalidP) free(nn->outputs);
    if (nn->activations != NN_invalidP) free(nn->activations);
    if (nn->deltas != NN_invalidP) free(nn->deltas);

    free(nn);
}








void trainNN(NeuralNetwork *nn, Matrix trainingData, unsigned int batchSize) {
    if (nn == NN_invalidP) throw "trainNN: invalid arguments";
    int trainCount = trainingData.rows;
    if (batchSize > trainCount) throw "trainNN: invalid arguments";
    for (int i = 0; i < trainCount; ++i) {
        Matrix input = getSubMatrix(trainingData, i, 0, 1, trainingData.cols - nn->numOutputs);
        Matrix expected = getSubMatrix(trainingData, i, trainingData.cols - nn->numOutputs, 1, nn->numOutputs);

        forward(*nn, input);
        backPropagation(*nn, expected);

        freeMatrix(input);
        freeMatrix(expected);

        if ((i + 1) % batchSize == 0 || i == trainingData.rows - 1) {
            updateWeightsAndBiases(*nn, batchSize);
        }
    }
}


void forward(NeuralNetwork nn, Matrix input) {
    if (nn.activations[0].elements != NN_invalidP) {
        freeMatrix(nn.activations[0]);
    }    
    nn.activations[0] = copyMatrix(input);

    for (int i = 1; i < nn.layerCount; ++i) {
        if (nn.activations[i].elements != NN_invalidP) {
            freeMatrix(
                nn.activations[i]);
        }
        if (nn.outputs[i].elements != NN_invalidP) {
            freeMatrix(nn.outputs[i]);
        }

        // compute new values
        Matrix result = multiplyMatrix(nn.activations[i - 1], nn.weights[i-1]);
        addMatrixInPlace(result, nn.biases[i-1]);

        nn.outputs[i] = copyMatrix(result);
        nn.activations[i] = applyActivation(nn, result, i);
        freeMatrix(result);
    }
}


void backPropagation(NeuralNetwork nn, Matrix expectedOutput) {
    // free existing deltas
    for (int i = 0; i < nn.layerCount-1; ++i) {
        if (nn.deltas[i].elements != NN_invalidP) {
            freeMatrix(nn.deltas[i]);
        }
    }

    nn.deltas[nn.layerCount - 2] = computeOutputLayerDeltas(nn, expectedOutput);

    // propagate error backward
    for (int i = nn.layerCount - 1; i > 0; --i) {
        Matrix transposedActivations = transposeMatrix(nn.activations[i - 1]);
        Matrix weightGradients = multiplyMatrix(transposedActivations, nn.deltas[i-1]);  // compute weight gradients for layer i Weight gradients = activations[i-1]^T * deltas[i-1]
        addMatrixInPlace(nn.weightsGradients[i-1], weightGradients);

        // free intermediate matrices
        freeMatrix(transposedActivations);
        freeMatrix(weightGradients);

        // compute bias gradients (sum over batch)
        Matrix biasGradients = copyMatrix(nn.deltas[i-1]);
        addMatrixInPlace(nn.biasesGradients[i-1], biasGradients);
        freeMatrix(biasGradients);

        // compute deltas for previous layer (if not input layer)
        if (i > 1) {
            Matrix transposedWeights = transposeMatrix(nn.weights[i-1]);
            Matrix inputGradients = multiplyMatrix(nn.deltas[i-1], transposedWeights);	// input gradients = deltas[i-1] * weights[i]^T
            freeMatrix(transposedWeights);

            Matrix activationDerivative = computeActivationDerivative(nn.outputs[i - 1], nn.config.hiddenLayersAF);	// activation derivative = f'(outputs[i-1])
            nn.deltas[i - 2] = multiplyMatrixElementWise(inputGradients, activationDerivative); 	// deltas for previous layer = inputGradients ⊙ activationDerivative
            // free intermediate matrices
            freeMatrix(inputGradients);
            freeMatrix(activationDerivative);
        }
    }
}

void updateWeightsAndBiases(NeuralNetwork nn, int batchSize) {
    // this loop can be parallelized
    for (int i = 0; i < nn.layerCount-1; ++i) {
        // update weights
        scaleMatrixInPlace(nn.weightsGradients[i], 1.0f / batchSize);
        Matrix scaledMatrix = scaleMatrix(nn.weightsGradients[i], nn.config.learningRate);
        subtractMatrixInPlace(nn.weights[i], scaledMatrix);
        freeMatrix(scaledMatrix);
        fillMatrix(nn.weightsGradients[i], 0.0f); // reset gradients

        // update biases
        scaleMatrixInPlace(nn.biasesGradients[i], 1.0f / batchSize);
        scaledMatrix = scaleMatrix(nn.biasesGradients[i], nn.config.learningRate);
        subtractMatrixInPlace(nn.biases[i], scaledMatrix);
        freeMatrix(scaledMatrix);
        fillMatrix(nn.biasesGradients[i], 0.0f); // reset gradients
    }
}


Matrix computeOutputLayerDeltas(NeuralNetwork nn, Matrix expectedOutput) {
    Matrix predicted = nn.activations[nn.layerCount - 1];
    Matrix rawPredicted = nn.outputs[nn.layerCount - 1];
    Matrix curLayerDeltas;
    if (nn.numOutputs > 1) { // multi output
        curLayerDeltas = computeMultipleOutputLossDerivativeMatrix(predicted, expectedOutput, nn.config.lossFunction); // because the af derivative and the loss derivative simplify each other only one calculation is needed
    }
    else { // single output
        Matrix dLoss_dY = computeLossDerivative(predicted, expectedOutput, nn.config.lossFunction); // derivative of the loss function
        Matrix activationDerivative = computeActivationDerivative(rawPredicted, nn.config.outputLayerAF);
        curLayerDeltas = multiplyMatrixElementWise(dLoss_dY, activationDerivative); // delta = dLoss_dY * derivative of the activation function with the non-activated output as input
        freeMatrix(dLoss_dY);
        freeMatrix(activationDerivative);
    }
    return curLayerDeltas;
}

Matrix computeMultipleOutputLossDerivativeMatrix(Matrix output, Matrix expectedOutput, int lf) {
    switch (lf) {
        case NN_CCE:
            return CCElossDerivativeMatrix(output, expectedOutput);
        default:
            throw "computeMultipleOutputLossDerivativeMatrix: Unsupported loss function: " + lf;
    }
}

Matrix computeLossDerivative(Matrix outputs, Matrix expectedOutputs, int lf) {
    Matrix derivative = createMatrix(outputs.rows, outputs.cols);
    for (int j = 0; j < outputs.cols; j++) {
        derivative.elements[j] = lossDerivative(outputs.elements[j], expectedOutputs.elements[j], lf);
    }
    return derivative;
}

float lossDerivative(float output, float expectedOutput, int lf) {
    switch (lf) {
    case NN_MSE:
        return MSElossDerivative(output, expectedOutput);
    case NN_BCE:
        return BCElossDerivative(output, expectedOutput);
    default:
        break;
    }
    return MSElossDerivative(output, expectedOutput);
}

Matrix computeActivationDerivative(Matrix outputs, int af) {
    Matrix derivative = createMatrix(outputs.rows, outputs.cols);
    for (int i = 0; i < outputs.rows; i++) {
        for (int j = 0; j < outputs.cols; j++) {
            derivative.elements[i * derivative.cols + j] = AFDerivative(outputs.elements[i * outputs.cols + j], af);
        }
    }
    return derivative;
}

float AFDerivative(float x, int af) {
    switch (af) {
    case NN_SIGMOID:
    {
        sigmoidDerivative(sigmoid(x));
    }
    case NN_RELU:
        return reluDerivative(x);
    default:
        break;
    }
    return 1;
}

Matrix applyActivation(NeuralNetwork nn, Matrix mat, int iLayer) {
    Matrix activated;

    if ((iLayer == nn.layerCount - 1) && (nn.numOutputs > 1)) { // try to apply non mutually exclusive multiple clases AFs first.
        activated = multipleOutputActivationFunction(mat, nn.config.outputLayerAF);
        if (activated.elements != NN_invalidP) {
            return activated;
        }
    }
    
    activated = createMatrix(mat.rows, mat.cols);
    // if they were not selected proceed with mutually exclusive AF.
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            if (iLayer == nn.layerCount - 1) {
                activated.elements[i * activated.cols + j] = activationFunction(mat.elements[i * mat.cols + j], nn.config.outputLayerAF);
            }
            else {
                activated.elements[i * activated.cols + j] = activationFunction(mat.elements[i * mat.cols + j], nn.config.hiddenLayersAF);
            }
        }
    }
    return activated;
}

Matrix multipleOutputActivationFunction(Matrix mat, int af) {
    Matrix res = {0,0,NN_invalidP};

    switch (af) {
        case NN_SOFTMAX:
            return softmax(mat);
        default:
            break;
    }
    return res;
}

float activationFunction(float x, int af) {
    switch (af) {
        case NN_SIGMOID:
            return sigmoid(x);
        case NN_RELU:
            return relu(x);
        default:
            break;
    }

    return x;
}

float computeAverageLossNN(NeuralNetwork *nn, Matrix trainingData) {
    if (nn == NN_invalidP) throw "computeAverageLossNN: invalid arguments";
    int numSamples = trainingData.rows;
    float totalLoss = 0.0;
    
    for (int i = 0; i < numSamples; i++) {
        Matrix input = getSubMatrix(trainingData, i, 0, 1, trainingData.cols - nn->numOutputs);
        Matrix expected = getSubMatrix(trainingData, i, trainingData.cols - nn->numOutputs, 1, nn->numOutputs);
        
        forward(*nn, input);
        Matrix prediction = nn->activations[nn->layerCount - 1];
        
        if(nn->numOutputs>1) {
            totalLoss+=multipleOutputLoss(prediction, expected, nn->config.lossFunction);
        }else {
            // calculate loss for this example
            for (int j = 0; j < nn->numOutputs; j++) {
                totalLoss+=loss(prediction.elements[j], expected.elements[j], nn->config.lossFunction);
            }
        }
        freeMatrix(input);
        freeMatrix(expected);
    }
    return totalLoss / numSamples;
}

float multipleOutputLoss(Matrix output, Matrix expectedOutput, int lf) {
    switch(lf) {
    case NN_CCE:
        return CCEloss(output,expectedOutput);
    default:
        throw "multipleOutputLoss: Unsupported loss function: " + lf;
           
    }
}

float loss(float output, float expectedOutput, int lf) {
    switch (lf) {
    case NN_MSE:
        return MSEloss(output, expectedOutput);
    case NN_BCE:
        return BCEloss(output, expectedOutput);
    default:
        break;
    }
    return MSEloss(output, expectedOutput);
}

float computeAccuracyNN(NeuralNetwork *nn, Matrix dataset) {
    if (nn == NN_invalidP) throw "computeAccuracyNN: invalid arguments";
    if(nn->numOutputs>1) {
        return computeMultiClassAccuracy(*nn, dataset);
    }else {
        return computeSingleOutputAccuracy(*nn, dataset);
    }
}

float computeMultiClassAccuracy(NeuralNetwork nn, Matrix dataset) {
    int correct = 0;
    for(int i=0; i<dataset.rows; i++) {
        Matrix input = getSubMatrix(dataset, i, 0, 1, dataset.cols - nn.numOutputs);
        Matrix output = getSubMatrix(dataset, i, dataset.cols - nn.numOutputs, 1, nn.numOutputs);
        
        forward(nn, input);
        Matrix pred = nn.activations[nn.layerCount-1];
        
        int predClass = 0;
        float maxVal = pred.elements[0];
        for(int j=1; j< nn.numOutputs; j++) {
            if(pred.elements[j] > maxVal) {
                maxVal = pred.elements[j];
                predClass = j;
            }
        }
        
        int trueClass = 0;
        for(int j=0; j< nn.numOutputs; j++) {
            if(output.elements[j] == 1.0) {
                trueClass = j;
                break;
            }
        }
        
        if(predClass == trueClass) correct++;
        freeMatrix(input);
        freeMatrix(output);
    }
    return (float)correct/dataset.rows*100;
}

float computeSingleOutputAccuracy(NeuralNetwork nn, Matrix dataset) {
    int numSamples = dataset.rows;
    int correct = 0;
    for (int i = 0; i < numSamples; i++) {
        Matrix input = getSubMatrix(dataset, i, 0, 1, dataset.cols - 1);
        Matrix expected = getSubMatrix(dataset, i, dataset.cols - 1, 1, 1);
        forward(nn, input);
        float prediction = nn.activations[nn.layerCount - 1].elements[0];
        int predictedLabel = (prediction >= 0.5) ? 1 : 0;
        int trueLabel = (int) expected.elements[0];
        if (predictedLabel == trueLabel) {
            correct++;
        }
        freeMatrix(input);
        freeMatrix(expected);
    }
    return (float) correct / numSamples * 100;
}

void saveStateNN(NeuralNetwork *nn){
    // !! TO DO implement saveStateNN
}

NeuralNetwork* loadStateNN(const char* filename) {
    // !! TO DO implement loadStateNN
}

Matrix predictNN(NeuralNetwork* nn, Matrix input) {
    forward(*nn, input);
    return nn->activations[nn->layerCount - 1];
}


void initializeMatrixRand(Matrix mat, float mean, float stddev) {
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            mat.elements[i * mat.cols + j] = randomNormal(mean, stddev);
        }
    }
}

float randomNormal(float mean, float stddev) {
    float u1 = (float)((float)rand() / (float)RAND_MAX);
    float u2 = (float)((float)rand() / (float)RAND_MAX);

    if (u1 < NN_epsilon) u1= NN_epsilon; // avoid log(0) errors

    float z0 = (float)(sqrt(-2.0f * log(u1)) * cos(2.0f * M_PI * u2));
    
    return mean + stddev * z0;
}


float relu(float x) {
    return 0 >= x ? 0 : x;
}

float reluDerivative(float x) {
    return x <= 0 ? (float)0 : (float)1;
}

float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float sigmoidDerivative(float sig) {
    return sig * (1.0f - sig);
}

Matrix softmax(Matrix mat) {
    Matrix result = createMatrix(mat.rows, mat.cols);

    for (int i = 0; i < mat.rows; i++) {
        float max = mat.elements[i * mat.cols];
        for (int j = 1; j < mat.cols; j++) {
            if (mat.elements[i * mat.cols + j] > max) {
                max = mat.elements[i * mat.cols + j];
            }
        }
        float sum = 0.0;
        for (int j = 0; j < mat.cols; j++) {
            result.elements[i * result.cols + j] = exp(mat.elements[i * mat.cols + j] - max);
            sum += result.elements[i * result.cols + j];
        }
        if (sum < NN_epsilon)sum += NN_epsilon; // avoid division by 0
        for (int j = 0; j < mat.cols; j++) {
            result.elements[i * result.cols + j] /= sum;
        }
    }
    return result;
}

float CCEloss(Matrix predictions, Matrix labels) {
    if (predictions.rows != labels.rows || predictions.cols != labels.cols) {
        throw "CCEloss: Predictions and labels must have the same dimensions.";
    }

    float loss = 0.0;
    for (int i = 0; i < predictions.rows; i++) {
        for (int j = 0; j < predictions.cols; j++) {
            float predicted = predictions.elements[i * predictions.cols + j];
            float expected = labels.elements[i * labels.cols + j];

            // add NN_epsilon to avoid log(0)
            predicted = (predicted < NN_epsilon) ? NN_epsilon : predicted;
            predicted = (predicted > 1.0f - NN_epsilon) ? 1.0f - NN_epsilon : predicted;

            // CCE formula: -sum(y * log(p))
            loss += expected * log(predicted);
        }
    }

    // average the loss over all samples
    return -loss / predictions.rows;
}

Matrix CCElossDerivativeMatrix(Matrix predicted, Matrix expected) {
    // for softmax + categorical cross-entropy, delta = predicted - true label
    Matrix deltas = subtractMatrix(predicted, expected);
    return deltas;
}

float BCEloss(float output, float expectedOutput) {
    // clip output to avoid log(0)
    output = NN_epsilon >= (1 - NN_epsilon <= output ? 1 - NN_epsilon : output) ? NN_epsilon : (1 - NN_epsilon <= output ? 1 - NN_epsilon : output);
    return -(expectedOutput * log(output) + (1 - expectedOutput) * log(1 - output));
}

float BCElossDerivative(float output, float expectedOutput) {
    // clip output to avoid division by zero
    output = NN_epsilon >= (1 - NN_epsilon <= output ? 1 - NN_epsilon : output) ? NN_epsilon : (1 - NN_epsilon <= output ? 1 - NN_epsilon : output);
    return (output - expectedOutput) / (output * (1 - output));
}

float MSEloss(float output, float expectedOutput) {
    float error = 0;
    error = (output - expectedOutput);
    error = error * error;
    return error;
}

float MSElossDerivative(float output, float expectedOutput) {
    float error = 0;
    error = output - expectedOutput;
    return error;
}