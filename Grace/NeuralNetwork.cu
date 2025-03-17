// check memory leaks
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>

#ifdef _DEBUG
#define malloc(s) _malloc_dbg(s, _NORMAL_BLOCK, __FILE__, __LINE__)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "NeuralNetwork.h"
#include "Matrix.h"
/**
* 
* default cases:
*   AF(single output) : none(f(x) = x)
*   AF DERIVATIVE(single output) : 1
*   LF(single output) : MSE
*   LF DERIVATIVE(single output) : MSE DERIVATIVE
*
*   AF(multiple output) : SOFTMAX
*   AF DERIVATIVE(multiple output) : CCE AND SOFTMAX DERIVATIVE
*   LF(multiple output) : CCE
*   LF(multiple output) : CCE AND SOFTMAX DERIVATIVE
* 
* TODO list:
*   write tests.
*   implement regularization.
*   implement optimizers.
*   improve documentation.
*   implement cuda kernels for gpu parallelied computations.
*   improve error handling.
*   implement a validate NN function.
*   
*/

struct NeuralNetwork {
    const int* architecture;
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

void forward(NeuralNetwork nn, Matrix input);
void backPropagation(NeuralNetwork nn, Matrix expectedOutput);
Matrix applyActivation(NeuralNetwork nn, Matrix matrix, unsigned int iLayer);
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
void updateWeightsAndBiases(NeuralNetwork nn, unsigned int batchSize);
float loss(float output, float expectedOutput, int lf);
float multipleOutputLoss(Matrix output, Matrix expectedOutput, int lf);
float CCEloss(Matrix predictions, Matrix labels);
float computeSingleOutputAccuracy(NeuralNetwork nn, Matrix dataset);
float computeMultiClassAccuracy(NeuralNetwork nn, Matrix dataset);
float randomNormal(float mean, float stddev);
float sigmoidDerivative(float sig);
NNStatus checkNNConfig(NNConfig config);
const char* NNStatusToString(NNStatus code);
float initializationFunction(NNInitializationFunction weightInitializerF, unsigned int nIn, unsigned int nOut);

NNStatus createNeuralNetwork(const int *architecture, const unsigned int layerCount, NNConfig config, NeuralNetwork **nnA) {
    if (nnA == NN_invalidP) return NN_ERROR_INVALID_ARGUMENT;
    NeuralNetwork *nn = (NeuralNetwork*) malloc(sizeof(NeuralNetwork));
    if (nn == NN_invalidP) return NN_ERROR_MEMORY_ALLOCATION;
    
    NNStatus err;
    err=checkNNConfig(config);
    if (err != 0)return err;

    nn->config = config;

    nn->architecture = architecture;
    nn->layerCount = layerCount;
    nn->weights = (Matrix*) malloc(sizeof(Matrix) * (layerCount-1)); if (nn->weights == NN_invalidP)return NN_ERROR_MEMORY_ALLOCATION;
    nn->biases = (Matrix*)malloc(sizeof(Matrix) * (layerCount-1)); if (nn->biases == NN_invalidP)return NN_ERROR_MEMORY_ALLOCATION;
    nn->weightsGradients = (Matrix*)malloc(sizeof(Matrix) * (layerCount-1)); if (nn->weightsGradients == NN_invalidP)return NN_ERROR_MEMORY_ALLOCATION;
    nn->biasesGradients = (Matrix*)malloc(sizeof(Matrix) * (layerCount-1)); if (nn->biasesGradients == NN_invalidP)return NN_ERROR_MEMORY_ALLOCATION;
    nn->outputs = (Matrix*)malloc(sizeof(Matrix) * layerCount); if (nn->outputs == NN_invalidP)return NN_ERROR_MEMORY_ALLOCATION;
    nn->activations = (Matrix*)malloc(sizeof(Matrix) * layerCount); if (nn->activations == NN_invalidP)return NN_ERROR_MEMORY_ALLOCATION;
    nn->numOutputs = architecture[layerCount - 1];

    /*
     * Allocating and initializing weights, biases, and gradients for layers 1 to layerCount-1.
     * initializing weights using the selected initialization method and biases to 0.01 (which helps prevent dead neurons).
     */
     
    srand((unsigned int)time(NULL));
    for (unsigned int i = 0; i < layerCount - 1; ++i) {
        nn->weights[i] = createMatrix(architecture[i], architecture[i + 1]);
        nn->biases[i] = createMatrix(1, architecture[i + 1]);
        nn->weightsGradients[i] = createMatrix(architecture[i], architecture[i + 1]);
        nn->biasesGradients[i] = createMatrix(1, architecture[i + 1]);

        // he initialization
        float stddev = initializationFunction(nn->config.weightInitializerF, nn->weights[i].rows, nn->weights[i].cols);
        float mean = 0.0f;
        initializeMatrixRand(nn->weights[i], mean, stddev);
        stddev = initializationFunction(nn->config.weightInitializerF, nn->biases[i].rows, nn->biases[i].cols);
        fillMatrix(nn->biases[i], 0.01f);

        // Initialize gradients to zero
        fillMatrix(nn->weightsGradients[i], 0);
        fillMatrix(nn->biasesGradients[i], 0);
        float sum = 0.0;
        for (unsigned int j = 0; j < nn->weights[i].rows * nn->weights[i].cols; ++j) {
            sum += fabs(nn->weights[i].elements[j]);
        }
    }

    for (unsigned int i = 0; i < layerCount; ++i) {
        nn->outputs[i] = { 0, 0, NN_invalidP };
        nn->activations[i] = { 0, 0, NN_invalidP };
    }

    *nnA = nn;
	return NN_OK;
}

NNStatus freeNeuralNetwork(NeuralNetwork *nn) {
    if (nn == NN_invalidP) return NN_ERROR_INVALID_ARGUMENT;
    // free all matrices inside matrices arrays
    for (unsigned int i = 0; i < nn->layerCount; ++i) {
        if (nn->outputs[i].elements != NN_invalidP) freeMatrix(nn->outputs[i]);
        if (nn->activations[i].elements != NN_invalidP) freeMatrix(nn->activations[i]);
    }
    for (unsigned int i = 0; i < nn->layerCount - 1; ++i) {
        if (nn->weights[i].elements != NN_invalidP) freeMatrix(nn->weights[i]);
        if (nn->biases[i].elements != NN_invalidP) freeMatrix(nn->biases[i]);
        if (nn->weightsGradients[i].elements != NN_invalidP) freeMatrix(nn->weightsGradients[i]);
        if (nn->biasesGradients[i].elements != NN_invalidP) freeMatrix(nn->biasesGradients[i]);
    }

    // free matrices arrays
    if (nn->weights != NN_invalidP) free(nn->weights);
    if (nn->biases != NN_invalidP) free(nn->biases);
    if (nn->weightsGradients != NN_invalidP) free(nn->weightsGradients);
    if (nn->biasesGradients != NN_invalidP) free(nn->biasesGradients);
    if (nn->outputs != NN_invalidP) free(nn->outputs);
    if (nn->activations != NN_invalidP) free(nn->activations);

    free(nn);
    return NN_OK;
}


NNStatus trainNN(NeuralNetwork *nn, Matrix trainingData, unsigned int batchSize) {
    if (nn == NN_invalidP) return NN_ERROR_INVALID_ARGUMENT;
    unsigned int trainCount = trainingData.rows;
    if (batchSize > trainCount) return NN_ERROR_INVALID_ARGUMENT;
    for (unsigned int i = 0; i < trainCount; ++i) {
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
    return NN_OK;
}

void forward(NeuralNetwork nn, Matrix input) {
    if (nn.activations[0].elements != NN_invalidP) {
        freeMatrix(nn.activations[0]);
    }    
    nn.activations[0] = copyMatrix(input);

    for (unsigned int i = 1; i < nn.layerCount; ++i) {
        if (nn.activations[i].elements != NN_invalidP) {
            freeMatrix(nn.activations[i]);
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
    Matrix deltas = computeOutputLayerDeltas(nn, expectedOutput);

    // propagate error backward
    for (unsigned int i = nn.layerCount - 1; i > 0; --i) {
        Matrix transposedActivations = transposeMatrix(nn.activations[i - 1]);
        Matrix weightGradients = multiplyMatrix(transposedActivations, deltas);  // compute weight gradients for layer, Weight gradients = activations[i-1]^T * deltas
        addMatrixInPlace(nn.weightsGradients[i - 1], weightGradients);

        // free intermediate matrices
        freeMatrix(transposedActivations);
        freeMatrix(weightGradients);

        // compute bias gradients
        Matrix biasGradients = copyMatrix(deltas);
        addMatrixInPlace(nn.biasesGradients[i - 1], biasGradients);
        freeMatrix(biasGradients);

        // compute deltas for previous layer (if not input layer)
        // delta[l] = (W[l+1]^T * delta[l+1]) ⊙ af'(z[l])
        if (i > 1) {
            Matrix transposedWeights = transposeMatrix(nn.weights[i - 1]);
            Matrix inputGradients = multiplyMatrix(deltas, transposedWeights);	// input gradients = deltas * weights[i]^T
            freeMatrix(transposedWeights);
            freeMatrix(deltas);

            Matrix activationDerivative = computeActivationDerivative(nn.outputs[i - 1], nn.config.hiddenLayersAF);	// activation derivative = f'(outputs[i-1])
            deltas = multiplyMatrixElementWise(inputGradients, activationDerivative); 	// deltas for previous layer = inputGradients ⊙ activationDerivative
            // free intermediate matrices
            freeMatrix(inputGradients);
            freeMatrix(activationDerivative);
        }
    }
    freeMatrix(deltas); // if (i > 1) skips deltas of first hidden layer
}

/*
void backPropagation(NeuralNetwork nn, Matrix expectedOutput) {
    // free existing deltas
    
    for (unsigned int i = 0; i < nn.layerCount-1; ++i) {
        if (nn.deltas[i].elements != NN_invalidP) {
            freeMatrix(nn.deltas[i]);
        }
    }
    
    // compute deltas for last layer.
    nn.deltas[nn.layerCount - 2] = computeOutputLayerDeltas(nn, expectedOutput);

    // propagate error backward
    for (unsigned int i = nn.layerCount - 1; i > 0; --i) {
        Matrix transposedActivations = transposeMatrix(nn.activations[i - 1]);
        Matrix weightGradients = multiplyMatrix(transposedActivations, nn.deltas[i-1]);  // compute weight gradients for layer, Weight gradients = activations[i-1]^T * deltas[i-1]
        addMatrixInPlace(nn.weightsGradients[i-1], weightGradients);

        // free intermediate matrices
        freeMatrix(transposedActivations);
        freeMatrix(weightGradients);

        // compute bias gradients
        Matrix biasGradients = copyMatrix(nn.deltas[i-1]);
        addMatrixInPlace(nn.biasesGradients[i-1], biasGradients);
        freeMatrix(biasGradients);

        // compute deltas for previous layer (if not input layer)
        // delta[l] = (W[l+1]^T * delta[l+1]) ⊙ af'(z[l])
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
*/
void updateWeightsAndBiases(NeuralNetwork nn, unsigned int batchSize) {
    // this loop can be parallelized
    for (unsigned int i = 0; i < nn.layerCount-1; ++i) {
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
        curLayerDeltas = multiplyMatrixElementWise(dLoss_dY, activationDerivative); // delta = dLoss_dY * derivative of the activation function with respect to the non-activated output
        freeMatrix(dLoss_dY);
        freeMatrix(activationDerivative);
    }
    return curLayerDeltas;
}

Matrix computeMultipleOutputLossDerivativeMatrix(Matrix output, Matrix expectedOutput, int lf) {
    switch (lf) {
        case NN_LOSS_CCE:
            return CCElossDerivativeMatrix(output, expectedOutput);
        default:
            return CCElossDerivativeMatrix(output, expectedOutput);
    }
}

Matrix computeLossDerivative(Matrix outputs, Matrix expectedOutputs, int lf) {
    Matrix derivative = createMatrix(outputs.rows, outputs.cols);
    for (unsigned int j = 0; j < outputs.cols; j++) {
        derivative.elements[j] = lossDerivative(outputs.elements[j], expectedOutputs.elements[j], lf);
    }
    return derivative;
}

float lossDerivative(float output, float expectedOutput, int lf) {
    switch (lf) {
    case NN_LOSS_MSE:
        return MSElossDerivative(output, expectedOutput);
    case NN_LOSS_BCE:
        return BCElossDerivative(output, expectedOutput);
    default:
        return MSElossDerivative(output, expectedOutput);
        break;
    }
}

Matrix computeActivationDerivative(Matrix outputs, int af) {
    Matrix derivative = createMatrix(outputs.rows, outputs.cols);
    for (unsigned int i = 0; i < outputs.rows; i++) {
        for (unsigned int j = 0; j < outputs.cols; j++) {
            derivative.elements[i * derivative.cols + j] = AFDerivative(outputs.elements[i * outputs.cols + j], af);
        }
    }
    return derivative;
}

float AFDerivative(float x, int af) {
    switch (af) {
    case NN_ACTIVATION_SIGMOID:
    {
        sigmoidDerivative(sigmoid(x));
    }
    case NN_ACTIVATION_RELU:
        return reluDerivative(x);
    default:
        return 1;
        break;
    }
}

Matrix applyActivation(NeuralNetwork nn, Matrix mat, unsigned int iLayer) {
    Matrix activated;

    if ((iLayer == nn.layerCount - 1) && (nn.numOutputs > 1)) { // try to apply non mutually exclusive multiple clases AFs first.
        activated = multipleOutputActivationFunction(mat, nn.config.outputLayerAF);
        if (activated.elements != NN_invalidP) {
            return activated;
        }
    }
    
    activated = createMatrix(mat.rows, mat.cols);
    // if they were not selected proceed with mutually exclusive AF.
    for (unsigned int i = 0; i < mat.rows; i++) {
        for (unsigned int j = 0; j < mat.cols; j++) {
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
        case NN_ACTIVATION_SOFTMAX:
            return softmax(mat);
        default:
            return softmax(mat);
            break;
    }
}

float activationFunction(float x, int af) {
    switch (af) {
        case NN_ACTIVATION_SIGMOID:
            return sigmoid(x);
        case NN_ACTIVATION_RELU:
            return relu(x);
        default:
            return x;
            break;
    }
}

NNStatus computeAverageLossNN(NeuralNetwork *nn, Matrix trainingData, float *averageLoss) {
    if (nn == NN_invalidP) return NN_ERROR_INVALID_ARGUMENT;
    unsigned int numSamples = trainingData.rows;
    float totalLoss = 0.0;
    
    for (unsigned int i = 0; i < numSamples; i++) {
        Matrix input = getSubMatrix(trainingData, i, 0, 1, trainingData.cols - nn->numOutputs);
        Matrix expected = getSubMatrix(trainingData, i, trainingData.cols - nn->numOutputs, 1, nn->numOutputs);
        
        forward(*nn, input);
        Matrix prediction = nn->activations[nn->layerCount - 1];
        
        if(nn->numOutputs>1) {
            totalLoss+=multipleOutputLoss(prediction, expected, nn->config.lossFunction);
        }else {
            // calculate loss for this example
            for (unsigned int j = 0; j < nn->numOutputs; j++) {
                totalLoss+=loss(prediction.elements[j], expected.elements[j], nn->config.lossFunction);
            }
        }
        freeMatrix(input);
        freeMatrix(expected);
    }
    *averageLoss = totalLoss / numSamples;

    return NN_OK;
}

float multipleOutputLoss(Matrix output, Matrix expectedOutput, int lf) {
    switch(lf) {
        case NN_LOSS_CCE:
            return CCEloss(output,expectedOutput);
        default:
            return CCEloss(output, expectedOutput);
    }
}

float loss(float output, float expectedOutput, int lf) {
    switch (lf) {
        case NN_LOSS_MSE:
            return MSEloss(output, expectedOutput);
        case NN_LOSS_BCE:
            return BCEloss(output, expectedOutput);
        default:
            return MSEloss(output, expectedOutput);
            break;
    }
}

NNStatus computeAccuracyNN(NeuralNetwork *nn, Matrix dataset, float* accuracy) {
    if (nn == NN_invalidP) return NN_ERROR_INVALID_ARGUMENT;
    if(nn->numOutputs>1) {
        *accuracy = computeMultiClassAccuracy(*nn, dataset);
    }else {
        *accuracy = computeSingleOutputAccuracy(*nn, dataset);
    }

    return NN_OK;
}

float computeMultiClassAccuracy(NeuralNetwork nn, Matrix dataset) {
    int correct = 0;
    for(unsigned int i=0; i<dataset.rows; i++) {
        Matrix input = getSubMatrix(dataset, i, 0, 1, dataset.cols - nn.numOutputs);
        Matrix output = getSubMatrix(dataset, i, dataset.cols - nn.numOutputs, 1, nn.numOutputs);
        
        forward(nn, input);
        Matrix pred = nn.activations[nn.layerCount-1];
        
        int predClass = 0;
        float maxVal = pred.elements[0];
        for(unsigned int j=1; j< nn.numOutputs; j++) {
            if(pred.elements[j] > maxVal) {
                maxVal = pred.elements[j];
                predClass = j;
            }
        }
        
        int trueClass = 0;
        for(unsigned int j=0; j< nn.numOutputs; j++) {
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
    unsigned int numSamples = dataset.rows;
    int correct = 0;
    for (unsigned int i = 0; i < numSamples; i++) {
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

NNStatus saveStateNN(NeuralNetwork *nn, const char* fileName){
    // TODO implement correctly

    return NN_OK;
}

NNStatus loadStateNN(const char* filename, NeuralNetwork* nn) {

    // TODO implement correctly

    return NN_OK;
}

NNStatus predictNN(NeuralNetwork* nn, Matrix input, Matrix* output) {
    forward(*nn, input);
    *output = copyMatrix(nn->activations[nn->layerCount - 1]);

    return NN_OK;
}

float initializationFunction(NNInitializationFunction weightInitializerF, unsigned int nIn, unsigned int nOut) {
    switch (weightInitializerF) {
        case NN_INITIALIZATION_0:
            return 0;
            break;
        case NN_INITIALIZATION_1:
            return 1;
            break;
        case NN_INITIALIZATION_SPARSE:
            return NN_epsilon;
            break;
        case NN_INITIALIZATION_HE:
            return sqrt(2.0f / nIn);
            break;
        case NN_INITIALIZATION_HE_UNIFORM:
            return sqrt(6.0f / nIn);
            break;
        case NN_INITIALIZATION_XAVIER:
            return sqrt(2.0f / nIn+nOut);
            break;
        case NN_INITIALIZATION_XAVIER_UNIFORM:
            return sqrt(6.0f / nIn + nOut);
            break;
        default: return sqrt(2.0f / nIn); // default to HE
    }
}

void initializeMatrixRand(Matrix mat, float mean, float stddev) {
    for (unsigned int i = 0; i < mat.rows; i++) {
        for (unsigned int j = 0; j < mat.cols; j++) {
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

    for (unsigned int i = 0; i < mat.rows; i++) {
        float max = mat.elements[i * mat.cols];
        for (unsigned int j = 1; j < mat.cols; j++) {
            if (mat.elements[i * mat.cols + j] > max) {
                max = mat.elements[i * mat.cols + j];
            }
        }
        float sum = 0.0;
        for (unsigned int j = 0; j < mat.cols; j++) {
            result.elements[i * result.cols + j] = exp(mat.elements[i * mat.cols + j] - max);
            sum += result.elements[i * result.cols + j];
        }
        if (sum < NN_epsilon)sum += NN_epsilon; // avoid division by 0
        for (unsigned int j = 0; j < mat.cols; j++) {
            result.elements[i * result.cols + j] /= sum;
        }
    }
    return result;
}

float CCEloss(Matrix predictions, Matrix labels) {
    if (predictions.rows != labels.rows || predictions.cols != labels.cols) {
        return -1;
    }

    float loss = 0.0;
    for (unsigned int i = 0; i < predictions.rows; i++) {
        for (unsigned int j = 0; j < predictions.cols; j++) {
            float predicted = predictions.elements[i * predictions.cols + j];
            float expected = labels.elements[i * labels.cols + j];

            // avoid log(0)
            predicted = (predicted < NN_epsilon) ? NN_epsilon : predicted;
            predicted = (predicted > 1.0f - NN_epsilon) ? 1.0f - NN_epsilon : predicted;

            loss += expected * log(predicted);
        }
    }
    // CCE formula: -sum(y * log(p))
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

NNStatus checkNNConfig(NNConfig config) {
    if (config.learningRate <= 0)return NN_ERROR_INVALID_LEARNING_RATE;

    switch (config.hiddenLayersAF){
        case NN_ACTIVATION_SIGMOID: break;
        case NN_ACTIVATION_RELU: break;
        case NN_ACTIVATION_SOFTMAX: break;
        default: return NN_ERROR_UNSUPPORTED_HIDDEN_AF;
    }
    switch (config.outputLayerAF) {
        case NN_ACTIVATION_SIGMOID: break;
        case NN_ACTIVATION_RELU: break;
        case NN_ACTIVATION_SOFTMAX: {
            if (config.lossFunction != NN_LOSS_CCE) return NN_ERROR_INVALID_LOSS_AF_COMBINATION;
            break;
        }
        default: return NN_ERROR_UNSUPPORTED_OUTPUT_AF;
    }

    switch (config.lossFunction) {
        case NN_LOSS_CCE: {
            if (config.outputLayerAF!= NN_ACTIVATION_SOFTMAX) return NN_ERROR_INVALID_LOSS_AF_COMBINATION;
            break;
        }
        case NN_LOSS_MSE: break;
        case NN_LOSS_BCE: break;
        default: return NN_ERROR_INVALID_LOSS_FUNCTION;
    }

    return NN_OK;
}


const char* NNStatusToString(NNStatus code) {
    switch (code) {
        case NN_OK:
            return "Function executed without problems";
            break;
        case NN_ERROR_INVALID_ARGUMENT:
            return "One of the arguments passed to the function is invalid";
            break;
        case NN_ERROR_MEMORY_ALLOCATION:
            return "An error occured while trying to allocate memory";
            break;
        case NN_ERROR_INVALID_CONFIG:
            return "The NNConfig settings are invalid";
            break;
        case NN_ERROR_INVALID_LEARNING_RATE:
            return "The selected learning rate is invalid";
            break;
        case NN_ERROR_UNSUPPORTED_HIDDEN_AF:
            return "The selected activation function for the hidden layer is invalid or unsupported";
            break;
        case NN_ERROR_UNSUPPORTED_OUTPUT_AF:
            return "The selected activation function for the output layer is invalid or unsupported";
            break;
        case NN_ERROR_INVALID_LOSS_FUNCTION:
            return "The selected loss function is invalid or unsupported";
            break;
        case NN_ERROR_INVALID_LOSS_AF_COMBINATION:
            return "The selected loss function and activation function combination is invalid or unsupported";
            break;
        case NN_ERROR_IO_FILE:
            return "An error occured while storing / loading NeuralNetwork struct.";
            break;
        default:
            return "Unknown error";
            break;
    }
}