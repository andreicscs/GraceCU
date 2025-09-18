// check memory leaks
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "NeuralNetworkDebug.h"
#include "NeuralNetwork.h"
#include "Matrix.h"
#include <string.h>
#include <stdbool.h>

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
*   LF DERIVATIVE (multiple output) : CCE AND SOFTMAX DERIVATIVE
* 
* TODO list:
*   improve tests
*   improve documentation, write a read me with complete api documnetation, and how to install and run the project.
*   implement data loading and processing functions, consider taking nn_config as param.
*   implement regularization.
*   implement optimizers.
*   implement cuda kernels for gpu parallelied computations.
*   
*/

// if the debug header is being used, don't define the struct twice
#ifndef NN_DEBUG
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

bool forward(NeuralNetwork nn, Matrix input);
bool backPropagation(NeuralNetwork nn, Matrix expectedOutput, Matrix input);
Matrix applyActivation(const NeuralNetwork nn, Matrix matrix, unsigned int iLayer);
Matrix multipleOutputActivationFunction(Matrix mat, int af);
float activationFunction(float x, int af);
Matrix computeOutputLayerDeltas(const NeuralNetwork nn, Matrix expectedOutput);
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
bool updateWeightsAndBiases(NeuralNetwork nn, unsigned int batchSize);
float loss(float output, float expectedOutput, int lf);
float multipleOutputLoss(Matrix output, Matrix expectedOutput, int lf);
float CCEloss(Matrix predictions, Matrix labels);
bool computeSingleOutputAccuracy(const NeuralNetwork nn, Matrix dataset, float* out);
bool computeMultiClassAccuracy(const NeuralNetwork nn, Matrix dataset, float* out);
float randomNormal(float mean, float stddev);
float sigmoidDerivative(float sig);
NNStatus checkNNConfig(NNConfig config);
const char* NNStatusToString(NNStatus code);
float initializationFunction(NNInitializationFunction weightInitializerF, unsigned int nIn, unsigned int nOut);
NNStatus allocateNNMatricesArrays(NeuralNetwork *nn);
NNStatus validateNNSettings(const NeuralNetwork *nn);


/*
* matrices are allocatoted and initialized here.
* matrices[0] rapresent the first hidden layer, matrices[layerCount-1] rapresent the output layer.
* each column in a layer rapresents a neuron.
* 
*/
NNStatus createNeuralNetwork(const unsigned int *architecture, const unsigned int layerCount, NNConfig config, NeuralNetwork **nnP) {
    NeuralNetwork *nn = (NeuralNetwork*) malloc(sizeof(NeuralNetwork));
    if (nn == NN_invalidP) return NN_ERROR_MEMORY_ALLOCATION;
    nn->activations = NN_invalidP;
    nn->architecture = NN_invalidP;
    nn->biases = NN_invalidP;
    nn->biasesGradients = NN_invalidP;
    nn->outputs = NN_invalidP;
    nn->weights = NN_invalidP;
    nn->weightsGradients = NN_invalidP;

    nn->architecture = (unsigned int*)malloc(layerCount * sizeof(unsigned int));
    if (nn->architecture == NN_invalidP) {
        freeNeuralNetwork(nn);
        return NN_ERROR_MEMORY_ALLOCATION;
    }
    // memory is copied to avoid storing the pointer directly which might cause errors if the original pointer is freed
    memcpy((void*)nn->architecture, architecture, layerCount * sizeof(unsigned int)); 

	nn->layerCount = layerCount - 1; // -1 because the input layer is not counted in the layerCount variable
    nn->numOutputs = nn->architecture[nn->layerCount];
    nn->config = config;
    
    NNStatus err = validateNNSettings(nn);
    if (err != NN_OK) {
        freeNeuralNetwork(nn);
        return err;
    }
    
    err=allocateNNMatricesArrays(nn);
    if (err != NN_OK) {
        freeNeuralNetwork(nn);
        return err;
    }
    
    /*
     * Allocating and initializing weights, biases, and gradients for layers 0 to layerCount-1.
     * initializing weights using the selected initialization method and biases to 0.01 (which helps prevent dead neurons).
     */
     
    for (unsigned int i = 0; i < nn->layerCount; ++i) {
        
        nn->outputs[i] = EMPTY_MATRIX;
        nn->activations[i] = EMPTY_MATRIX;

        nn->weights[i] = createMatrix(architecture[i], architecture[i + 1]); 
        if (nn->weights[i].elements == MATRIX_invalidP) { freeNeuralNetwork(nn); return NN_ERROR_MEMORY_ALLOCATION; }

        nn->biases[i] = createMatrix(1, architecture[i + 1]); 
        if (nn->biases[i].elements == MATRIX_invalidP) { freeNeuralNetwork(nn); return NN_ERROR_MEMORY_ALLOCATION; }

        nn->weightsGradients[i] = createMatrix(architecture[i], architecture[i + 1]); 
        if (nn->weightsGradients[i].elements == MATRIX_invalidP) { freeNeuralNetwork(nn); return NN_ERROR_MEMORY_ALLOCATION; }

        nn->biasesGradients[i] = createMatrix(1, architecture[i + 1]); 
        if (nn->biasesGradients[i].elements == MATRIX_invalidP) { freeNeuralNetwork(nn); return NN_ERROR_MEMORY_ALLOCATION; }

        float stddev = initializationFunction(nn->config.weightInitializerF, nn->weights[i].rows, nn->weights[i].cols);
        float mean = 0.0f;
        initializeMatrixRand(nn->weights[i], mean, stddev);
        fillMatrix(nn->biases[i], 0.01f);
        // Initialize gradients to zero
        fillMatrix(nn->weightsGradients[i], 0);
        fillMatrix(nn->biasesGradients[i], 0);
    }

    *nnP = nn;
	return NN_OK;
}

NNStatus validateNNSettings(const NeuralNetwork *nn) {
    if (nn == NN_invalidP) return NN_ERROR_INVALID_ARGUMENT;
    if (nn->architecture == NN_invalidP || nn->layerCount < 1) {
        return NN_ERROR_INVALID_ARGUMENT;
    }
	for (unsigned int i = 0; i < nn->layerCount + 1; ++i) { //take in considration every layer including input and output
        if (nn->architecture[i] < 1) return NN_ERROR_INVALID_ARGUMENT;
    }

    NNStatus err;
    err = checkNNConfig(nn->config);
    if (err != NN_OK) {
        return err;
    }

    return NN_OK;
}

NNStatus allocateNNMatricesArrays(NeuralNetwork* nn) {
    nn->weights = NN_invalidP;
    nn->biases = NN_invalidP;
    nn->weightsGradients = NN_invalidP;
    nn->biasesGradients = NN_invalidP;
    nn->outputs = NN_invalidP;
    nn->activations = NN_invalidP;


    nn->weights = (Matrix*)malloc(sizeof(Matrix) * (nn->layerCount));
    if (nn->weights == NN_invalidP) { freeNeuralNetwork(nn); return NN_ERROR_MEMORY_ALLOCATION; }

    nn->biases = (Matrix*)malloc(sizeof(Matrix) * (nn->layerCount));
    if (nn->biases == NN_invalidP) { freeNeuralNetwork(nn); return NN_ERROR_MEMORY_ALLOCATION; }

    nn->weightsGradients = (Matrix*)malloc(sizeof(Matrix) * (nn->layerCount));
    if (nn->weightsGradients == NN_invalidP) { freeNeuralNetwork(nn); return NN_ERROR_MEMORY_ALLOCATION; }

    nn->biasesGradients = (Matrix*)malloc(sizeof(Matrix) * (nn->layerCount));
    if (nn->biasesGradients == NN_invalidP) { freeNeuralNetwork(nn); return NN_ERROR_MEMORY_ALLOCATION; }

    nn->outputs = (Matrix*)malloc(sizeof(Matrix) * nn->layerCount);
    if (nn->outputs == NN_invalidP) { freeNeuralNetwork(nn); return NN_ERROR_MEMORY_ALLOCATION; }

    nn->activations = (Matrix*)malloc(sizeof(Matrix) * nn->layerCount);
    if (nn->activations == NN_invalidP) { freeNeuralNetwork(nn); return NN_ERROR_MEMORY_ALLOCATION; }

    return NN_OK;
}

NNStatus freeNeuralNetwork(NeuralNetwork *nn) {
    if (nn == NN_invalidP) return NN_ERROR_INVALID_ARGUMENT;
    // free all matrices inside matrices arrays
    
    for (unsigned int i = 0; i < nn->layerCount; ++i) {
        if (nn->outputs != NN_invalidP) if (nn->outputs[i].elements != MATRIX_invalidP) freeMatrix(nn->outputs[i]); // check if the pointer itself is null first.
        if (nn->activations != NN_invalidP) if (nn->activations[i].elements != MATRIX_invalidP) freeMatrix(nn->activations[i]);
        if (nn->weights != NN_invalidP) if (nn->weights[i].elements != MATRIX_invalidP) freeMatrix(nn->weights[i]);
        if (nn->biases != NN_invalidP) if (nn->biases[i].elements != MATRIX_invalidP) freeMatrix(nn->biases[i]);
        if (nn->weightsGradients != NN_invalidP) if (nn->weightsGradients[i].elements != MATRIX_invalidP) freeMatrix(nn->weightsGradients[i]);
        if (nn->biasesGradients != NN_invalidP) if (nn->biasesGradients[i].elements != MATRIX_invalidP) freeMatrix(nn->biasesGradients[i]);
    }

    // free matrices arrays{
    if (nn->weights != NN_invalidP) { free(nn->weights); nn->weights = NN_invalidP; }
    if (nn->biases != NN_invalidP) { free(nn->biases); nn->biases = NN_invalidP; }
    if (nn->weightsGradients != NN_invalidP) { free(nn->weightsGradients); nn->weightsGradients = NN_invalidP; }
    if (nn->biasesGradients != NN_invalidP) { free(nn->biasesGradients); nn->biasesGradients = NN_invalidP; }
    if (nn->outputs != NN_invalidP) { free(nn->outputs); nn->outputs = NN_invalidP; }
    if (nn->activations != NN_invalidP) { free(nn->activations); nn->activations = NN_invalidP; }
    if (nn->architecture != NN_invalidP) { free((void*)nn->architecture); nn->architecture = NN_invalidP; }


    free(nn);
    nn = NN_invalidP;
    return NN_OK;
}

/*
* slow mini batches implementation is used to improve code clarity and maintainability
* would need to rewrite forward/backPropagation to take multiple samples at once.
*/
NNStatus trainNN(NeuralNetwork *nn, Matrix trainingData, unsigned int batchSize) {
    if (nn == NN_invalidP) return NN_ERROR_INVALID_ARGUMENT;
    if ((trainingData.cols- nn->numOutputs)!=nn->architecture[0]) return NN_ERROR_INVALID_ARGUMENT;
    if ((trainingData.cols - nn->architecture[0]) != nn->numOutputs) return NN_ERROR_INVALID_ARGUMENT;
    unsigned int trainCount = trainingData.rows;
    if (batchSize > trainCount || batchSize==0) return NN_ERROR_INVALID_ARGUMENT;
    for (unsigned int i = 0; i < trainCount; ++i) {
        Matrix input = getSubMatrix(trainingData, i, 0, 1, trainingData.cols - nn->numOutputs);
        if (input.elements==MATRIX_invalidP) return NN_ERROR_MEMORY_ALLOCATION;
        Matrix expected = getSubMatrix(trainingData, i, trainingData.cols - nn->numOutputs, 1, nn->numOutputs);
        if (expected.elements == MATRIX_invalidP) {
            freeMatrix(input);
            return NN_ERROR_MEMORY_ALLOCATION;
        }

        if (!forward(*nn, input)) {
            freeMatrix(input);
            freeMatrix(expected);
            return NN_ERROR_MEMORY_ALLOCATION;
        }
        if (!backPropagation(*nn, expected, input)) {
            freeMatrix(input);
            freeMatrix(expected);
            return NN_ERROR_MEMORY_ALLOCATION;
        }

        freeMatrix(input);
        freeMatrix(expected);

        if (i == trainCount - 1) { // if traincount is not a multiple of batchsize.
            if (!updateWeightsAndBiases(*nn, ((trainCount-1)%batchSize)+1))return NN_ERROR_MEMORY_ALLOCATION;
        }else if ((i + 1) % batchSize == 0) {
            if (!updateWeightsAndBiases(*nn, batchSize))return NN_ERROR_MEMORY_ALLOCATION;
        }
    }
    
    return NN_OK;
}

bool forward(NeuralNetwork nn, Matrix input) {    
    // first iteration outside of loop to initialize algorithm with the input.

    // if the matrix is allocated it gets freed, it will be replaced later.
    if (nn.activations[0].elements != MATRIX_invalidP) {
        freeMatrix(nn.activations[0]);
    }
    if (nn.outputs[0].elements != MATRIX_invalidP) {
        freeMatrix(nn.outputs[0]);
    }

    Matrix result = multiplyMatrix(input, nn.weights[0]);
    if (result.elements == MATRIX_invalidP) return false;
    addMatrixInPlace(result, nn.biases[0]);

    nn.outputs[0] = copyMatrix(result);
    if (nn.outputs[0].elements == MATRIX_invalidP) {
        freeMatrix(result);
        return false;
    }

    nn.activations[0] = applyActivation(nn, result, 1); // first hidden layer's activation because layer 0 is the input layer.
    if (nn.activations[0].elements == MATRIX_invalidP) {
        freeMatrix(result);
        return false;
    }
    freeMatrix(result);
    
    for (unsigned int i = 1; i < nn.layerCount; ++i) {
        if (nn.activations[i].elements != MATRIX_invalidP) {
            freeMatrix(nn.activations[i]);
        }
        if (nn.outputs[i].elements != MATRIX_invalidP) {
            freeMatrix(nn.outputs[i]);
        }
        
        // compute new values
        Matrix result = multiplyMatrix(nn.activations[i - 1], nn.weights[i]);
        if (result.elements == MATRIX_invalidP) return false;
        addMatrixInPlace(result, nn.biases[i]);

        nn.outputs[i] = copyMatrix(result);
        if (nn.outputs[i].elements == MATRIX_invalidP) {
            freeMatrix(result);
            return false;
        }

        nn.activations[i] = applyActivation(nn, result, i+1);
        if (nn.activations[i].elements == MATRIX_invalidP) {
            freeMatrix(result);
            return false;
        }
        freeMatrix(result);
    }
    return true;
}


/*
    Backpropagation gradient formulas:

    ∂L/∂W[l] = (a[l-1])^T · δ[l]
    ∂L/∂b[l] = δ[l]

    Explanation:

    Chain rule definition of δ:
        δ[l] = ∂L/∂z[l] = (∂L/∂a[l]) ⊙ af'(z[l])
        - The loss depends on activations a[l] = af(z[l])
        - So we must include the derivative of the activation af'(z[l])

       **  Why element-wise multiplication?
            Because z[l] is a vector, and each activation af(z[l]_j)
                depends only on its own pre-activation z[l]_j, not on others.
        
        How is (∂L/∂a[l]) calculated?
            - For the output layer, it is computed directly from the loss function.
            - For hidden layers, it is computed from the next layer's δ[l+1] and weights W[l+1]:
			  ∂L/∂a[l] = δ[l+1] · W[l+1]^T (δ[l+1] already includes the activation derivative at layer l+1).

            ** why transpose W[l+1]?
                - Each neuron in layer l affects all neurons in layer l+1.
                - The derivative of the loss w.r.t. a[l]_j sums contributions from all neurons in layer l+1.
				- The transpose ensures that when multiplying δ[l+1] by W[l+1]^T, 
                    each neuron in layer l receives the weighted sum of errors from all neurons it feeds into in layer l+1

    1) Weight gradient:
        ∂L/∂W[l] = (a[l-1])^T · δ[l]
        ** Why?:
            ∂L/∂W[l] = ∂L/∂z[l] · ∂z[l]/∂W[l]
                        = δ[l] · (a[l-1])^T

	    ** Why is activations transposed?:
		    - Each weight connects one input neuron (from layer l-1) to one output neuron (in layer l), 
                so the gradient for a weight must combine:
                    (error of the output neuron) × (activation of the input neuron).
				To form all pairwise combinations (each output error × each input activation) 
                we need an outer product: δ[l] × (a[l-1])^T

    2) Bias gradient:
        ∂L/∂b[l] = δ[l]
        Because ∂z[l]/∂b[l] = 1.
        ** Why?:
            - Each bias b[l]_j affects only its own pre-activation z[l]_j. 
                so the derivative of z[l]_j w.r.t. b[l]_j is 1, and 0 for others.


    3) Propagation backwards:
        To send the error to the previous layer:
        δ[l-1] = δ[l] · W[l]^T ⊙ af'(z[l-1])

    Where:
    - L = Loss function
    - W[l] = weight matrix at layer l
    - b[l] = bias vector at layer l
    - a[l-1] = activations (outputs) from previous layer (l-1)
    - z[l] = pre-activation inputs = W[l]·a[l-1] + b[l]
    - af = activation function
    - af'(z[l]) = derivative of the activation function
    - δ[l] = "partialDerivative or error" at layer l, i.e. how much this layer contributed to the loss
	- ⊙ = element-wise multiplication
*/
bool backPropagation(NeuralNetwork nn, Matrix expectedOutput, Matrix input) {
    //compute output partial derivatives which uses output delta.
    Matrix partialDerivatives = computeOutputLayerDeltas(nn, expectedOutput);
    if (partialDerivatives.elements == MATRIX_invalidP) return false;

	// starting from the output layer, apply chainrule to compute gradients for each layer and partial gradients for the previous layer.
	for (int i = nn.layerCount - 1; i >= 0; --i) {
        Matrix transposedActivations;
        if (i == 0) {
			transposedActivations = transposeMatrix(input); // use input instead of nn.activations[-1] for the first hidden layer
        }
        else {
            transposedActivations = transposeMatrix(nn.activations[i - 1]); // to match dimensions, transpose a[l-1] so the multiplication gives a matrix shaped(input_dim × output_dim), same as W[l].
        }
        if (transposedActivations.elements == MATRIX_invalidP) {
            freeMatrix(partialDerivatives);
            return false;
        }

        Matrix weightGradients = multiplyMatrix(transposedActivations, partialDerivatives);  // compute weight gradients for each layer, Weight gradients = activations[i-1]^T * partialDerivatives (∂L/∂W[l] = (a[l-1])^T · δ[l])
        if (weightGradients.elements == MATRIX_invalidP) {
            freeMatrix(partialDerivatives);
            freeMatrix(transposedActivations);
            return false;
        }

		addMatrixInPlace(nn.weightsGradients[i], weightGradients); // sum the gradients over the batch

        // free intermediate matrices
        freeMatrix(transposedActivations);
        freeMatrix(weightGradients);

        // compute bias gradients
        addMatrixInPlace(nn.biasesGradients[i], partialDerivatives); // Bias gradients = partialGradient[i] (∂L/∂b[l] = δ[l]), sum the gradients over the batch
        
        // compute partialDerivatives for previous layer (if not input layer)
        // δ[i-1] = δ[i] * W[i]^T ⊙ f'(z[i-1])
        // so partialDerivatives[l-1] = W[l]^T * partialDerivatives[l] ⊙ af'(z[l-1])
        if (i > 0) {
            Matrix transposedWeights = transposeMatrix(nn.weights[i]);
            if (transposedWeights.elements == MATRIX_invalidP) {
                freeMatrix(partialDerivatives);
                return false;
            }

            Matrix inputGradients = multiplyMatrix(partialDerivatives, transposedWeights);	// input gradients = partialDerivatives * weights[i]^T (δ[i] * W[i]^T)
            if (inputGradients.elements == MATRIX_invalidP) {
                freeMatrix(partialDerivatives);
                freeMatrix(transposedWeights);
                return false;
            }

            freeMatrix(transposedWeights);
            freeMatrix(partialDerivatives);

            Matrix activationDerivative = computeActivationDerivative(nn.outputs[i - 1], nn.config.hiddenLayersAF);	// activation derivative = af'(outputs[i-1]) (f'(z[i-1]))
            if (activationDerivative.elements == MATRIX_invalidP) {
                freeMatrix(inputGradients);
                return false;
            }

            partialDerivatives = multiplyMatrixElementWise(inputGradients, activationDerivative); 	// partialDerivatives for previous layer = inputGradients ⊙ activationDerivative (δ[i] * W[i]^T ⊙ f'(z[i-1]))
            if (partialDerivatives.elements == MATRIX_invalidP) {
                freeMatrix(inputGradients);
                freeMatrix(activationDerivative);
                return false;
            }

            // free intermediate matrices
            freeMatrix(inputGradients);
            freeMatrix(activationDerivative);
        }
    }
    freeMatrix(partialDerivatives);
    return true;
}

bool updateWeightsAndBiases(NeuralNetwork nn, unsigned int batchSize) {
    for (unsigned int i = 0; i < nn.layerCount; ++i) {
        // update weights
        scaleMatrixInPlace(nn.weightsGradients[i], 1.0f / batchSize);
        Matrix scaledMatrix = scaleMatrix(nn.weightsGradients[i], nn.config.learningRate);
        if (scaledMatrix.elements == MATRIX_invalidP) return false;
        subtractMatrixInPlace(nn.weights[i], scaledMatrix);
        freeMatrix(scaledMatrix);
        fillMatrix(nn.weightsGradients[i], 0.0f); // reset gradients

        // update biases
        scaleMatrixInPlace(nn.biasesGradients[i], 1.0f / batchSize);
        scaledMatrix = scaleMatrix(nn.biasesGradients[i], nn.config.learningRate);
        if (scaledMatrix.elements == MATRIX_invalidP) return false;
        subtractMatrixInPlace(nn.biases[i], scaledMatrix);
        freeMatrix(scaledMatrix);
        fillMatrix(nn.biasesGradients[i], 0.0f); // reset gradients
    }
    return true;
}

Matrix computeOutputLayerDeltas(const NeuralNetwork nn, Matrix expectedOutput) {
    Matrix predicted = nn.activations[nn.layerCount - 1];
    Matrix rawPredicted = nn.outputs[nn.layerCount - 1];
    Matrix partialDerivatives;
    if (nn.numOutputs > 1) { // multi output
        partialDerivatives = computeMultipleOutputLossDerivativeMatrix(predicted, expectedOutput, nn.config.lossFunction); // because the af derivative and the loss derivative simplify each other using NN_LOSS_CCE and NN_ACTIVATION_SOFTMAX only delta is needed, needs to be revisited once other loss functions will be implemented
        if (partialDerivatives.elements == MATRIX_invalidP) return EMPTY_MATRIX ;
    }
    else { // single output
        Matrix dLoss_dY = computeLossDerivative(predicted, expectedOutput, nn.config.lossFunction); // derivative of the loss function
        if (dLoss_dY.elements == MATRIX_invalidP) return EMPTY_MATRIX ;
        
        Matrix activationDerivative = computeActivationDerivative(rawPredicted, nn.config.outputLayerAF);
        if (activationDerivative.elements == MATRIX_invalidP) {
            freeMatrix(dLoss_dY);
            return EMPTY_MATRIX ;
        }

        partialDerivatives = multiplyMatrixElementWise(dLoss_dY, activationDerivative); // partialGradient = dLoss_dY * derivative of the activation function with respect to the non-activated output
        if (partialDerivatives.elements == MATRIX_invalidP) {
            freeMatrix(dLoss_dY);
            freeMatrix(activationDerivative);
            return EMPTY_MATRIX ;
        }

        freeMatrix(dLoss_dY);
        freeMatrix(activationDerivative);
    }
    return partialDerivatives;
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
    if (derivative.elements == MATRIX_invalidP) return EMPTY_MATRIX ;

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
    if (derivative.elements == MATRIX_invalidP) return EMPTY_MATRIX ;
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
        return sigmoidDerivative(sigmoid(x));
    }
    case NN_ACTIVATION_RELU:
        return reluDerivative(x);
    default:
        return 1;
        break;
    }
}

Matrix applyActivation(const NeuralNetwork nn, Matrix mat, unsigned int iLayer) {
    Matrix activated;

    if ((iLayer == nn.layerCount) && (nn.numOutputs > 1)) { // try to apply non mutually exclusive multiple clases AFs first.
        activated = multipleOutputActivationFunction(mat, nn.config.outputLayerAF);
        if (activated.elements != MATRIX_invalidP) return activated;
    }
    
    activated = createMatrix(mat.rows, mat.cols);
    if (activated.elements == MATRIX_invalidP) return EMPTY_MATRIX ;

    // if they were not selected proceed with mutually exclusive AF.
    for (unsigned int i = 0; i < mat.rows; i++) {
        for (unsigned int j = 0; j < mat.cols; j++) {
            if (iLayer == nn.layerCount) {
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

NNStatus computeAverageLossNN(const NeuralNetwork *nn, Matrix trainingData, float *averageLoss) {
    if (nn == NN_invalidP) return NN_ERROR_INVALID_ARGUMENT;
    if (averageLoss == NN_invalidP) return NN_ERROR_INVALID_ARGUMENT;
    if ((trainingData.cols - nn->numOutputs) != nn->architecture[0]) return NN_ERROR_INVALID_ARGUMENT;
    if ((trainingData.cols - nn->architecture[0]) != nn->numOutputs) return NN_ERROR_INVALID_ARGUMENT;


    unsigned int numSamples = trainingData.rows;
    float totalLoss = 0.0;
    
    for (unsigned int i = 0; i < numSamples; i++) {
        Matrix input = getSubMatrix(trainingData, i, 0, 1, trainingData.cols - nn->numOutputs);
        if (input.elements == MATRIX_invalidP) return NN_ERROR_MEMORY_ALLOCATION;

        Matrix expected = getSubMatrix(trainingData, i, trainingData.cols - nn->numOutputs, 1, nn->numOutputs);
        if (expected.elements == MATRIX_invalidP) {
            freeMatrix(input);
            return NN_ERROR_MEMORY_ALLOCATION;
        }
        
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

NNStatus computeAccuracyNN(const NeuralNetwork *nn, Matrix dataset, float* accuracy) {
    if (nn == NN_invalidP) return NN_ERROR_INVALID_ARGUMENT;
    if (accuracy == NN_invalidP) return NN_ERROR_INVALID_ARGUMENT;
    if ((dataset.cols - nn->numOutputs) != nn->architecture[0]) return NN_ERROR_INVALID_ARGUMENT;
    if ((dataset.cols - nn->architecture[0]) != nn->numOutputs) return NN_ERROR_INVALID_ARGUMENT;


    if(nn->numOutputs>1) {
        if (!computeMultiClassAccuracy(*nn, dataset, accuracy))return NN_ERROR_MEMORY_ALLOCATION;
    }else {
        if (!computeSingleOutputAccuracy(*nn, dataset, accuracy)) return NN_ERROR_MEMORY_ALLOCATION;
    }

    return NN_OK;
}

bool computeMultiClassAccuracy(NeuralNetwork nn, Matrix dataset, float* out) {
    int correct = 0;
    for(unsigned int i=0; i<dataset.rows; i++) {
        Matrix input = getSubMatrix(dataset, i, 0, 1, dataset.cols - nn.numOutputs);
        if (input.elements == MATRIX_invalidP) return false;

        Matrix output = getSubMatrix(dataset, i, dataset.cols - nn.numOutputs, 1, nn.numOutputs);
        if (output.elements == MATRIX_invalidP) {
            freeMatrix(input);
            return false;
        }
        
        if (!forward(nn, input)) {
            freeMatrix(input);
            freeMatrix(output);
            return false;
        }
        
        Matrix pred = nn.activations[nn.layerCount - 1];
        
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
    *out = (float)correct/dataset.rows*100;
    return true;
}

bool computeSingleOutputAccuracy(NeuralNetwork nn, Matrix dataset, float* out) {
    unsigned int numSamples = dataset.rows;
    int correct = 0;
    for (unsigned int i = 0; i < numSamples; i++) {
        Matrix input = getSubMatrix(dataset, i, 0, 1, dataset.cols - 1);
        if (input.elements == MATRIX_invalidP) return false;

        Matrix expected = getSubMatrix(dataset, i, dataset.cols - 1, 1, 1);
        if (expected.elements == MATRIX_invalidP) {
            freeMatrix(input);
            return false;
        }

        if (!forward(nn, input)) {
            freeMatrix(input);
            freeMatrix(expected);
            return false;
        }

        float prediction = nn.activations[nn.layerCount - 1].elements[0];
        int predictedLabel = (prediction >= 0.5) ? 1 : 0;
        int trueLabel = (int) expected.elements[0];
        if (predictedLabel == trueLabel) {
            correct++;
        }
        freeMatrix(input);
        freeMatrix(expected);
    }
    *out= (float) correct / numSamples * 100;
    return true;
}

NNStatus saveStateNN(const NeuralNetwork *nn, FILE* fpOut){
    if (fpOut == NN_invalidP) {
        return NN_ERROR_INVALID_ARGUMENT;
    }

    size_t writtenElements = fwrite(&nn->layerCount, sizeof(unsigned int), 1, fpOut);
    if (writtenElements != 1) return NN_ERROR_IO_FILE;
    
	writtenElements = fwrite(nn->architecture, sizeof(unsigned int), nn->layerCount + 1, fpOut); // +1 for input layer
    if (writtenElements != nn->layerCount + 1) return NN_ERROR_IO_FILE;

    writtenElements = fwrite(&nn->config, sizeof(NNConfig), 1, fpOut);
    if (writtenElements != 1) return NN_ERROR_IO_FILE;

    for (unsigned int i = 0; i < nn->layerCount; ++i) {
        if (!storeMatrix(nn->weights[i], fpOut)) return NN_ERROR_IO_FILE;
        if (!storeMatrix(nn->biases[i], fpOut)) return NN_ERROR_IO_FILE;
        if (!storeMatrix(nn->weightsGradients[i], fpOut)) return NN_ERROR_IO_FILE;
        if (!storeMatrix(nn->biasesGradients[i], fpOut)) return NN_ERROR_IO_FILE;
        if (!storeMatrix(nn->outputs[i], fpOut)) return NN_ERROR_IO_FILE;
        if (!storeMatrix(nn->activations[i], fpOut)) return NN_ERROR_IO_FILE;
    }
    return NN_OK;
}

NNStatus loadStateNN(FILE* fpIn, NeuralNetwork** nnP) {
    if (fpIn == NN_invalidP) {
        return NN_ERROR_INVALID_ARGUMENT;
    }
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    if (nn == NN_invalidP) return NN_ERROR_MEMORY_ALLOCATION;

    unsigned int layerCount;
    size_t readElements = fread(&layerCount, sizeof(unsigned int), 1, fpIn);
    if (readElements != 1) return NN_ERROR_IO_FILE;
    
	unsigned int* architecture = (unsigned int*)malloc((layerCount + 1) * sizeof(unsigned int)); // +1 for input layer
    if (architecture == NN_invalidP) return NN_ERROR_MEMORY_ALLOCATION;
    readElements = fread(architecture, sizeof(unsigned int), layerCount + 1, fpIn);
    if (readElements != layerCount + 1) {
        free(architecture);
        architecture = NN_invalidP;
        return NN_ERROR_IO_FILE;
    }

    NNConfig config;
    readElements = fread(&config, sizeof(NNConfig), 1, fpIn);
    if (readElements != 1) {
        free(architecture);
        architecture = NN_invalidP;
        return NN_ERROR_IO_FILE;
    }
    
	nn->architecture = (unsigned int*)malloc((layerCount + 1) * sizeof(unsigned int)); // +1 for input layer
    if (nn->architecture == NN_invalidP) {
        free(architecture);
        architecture = NN_invalidP;
        freeNeuralNetwork(nn);
        return NN_ERROR_MEMORY_ALLOCATION;
    }
    // memory is copied to avoid storing the pointer directly which might cause errors if the original pointer is freed
	memcpy((void*)nn->architecture, architecture, (layerCount + 1) * sizeof(unsigned int));  // +1 for input layer
    free(architecture);
    architecture = NN_invalidP;
    nn->layerCount = layerCount;
    nn->numOutputs = nn->architecture[nn->layerCount];
    nn->config = config;

    NNStatus err = validateNNSettings(nn);
    if (err != NN_OK) {
        freeNeuralNetwork(nn);
        return err;
    }
    
    err = allocateNNMatricesArrays(nn);
    if (err != NN_OK) {
        freeNeuralNetwork(nn);
        return err;
    }

    for (unsigned int i = 0; i < nn->layerCount; ++i) {
        if (!loadMatrix(fpIn, &nn->weights[i])) {
            freeNeuralNetwork(nn);
            return NN_ERROR_IO_FILE; // could also return error becouse of failed malloc.
        }
        if (!loadMatrix(fpIn, &nn->biases[i])) {
            freeNeuralNetwork(nn);
            return NN_ERROR_IO_FILE;
        }
        if (!loadMatrix(fpIn, &nn->weightsGradients[i])) {
            freeNeuralNetwork(nn);
            return NN_ERROR_IO_FILE;
        }
        if (!loadMatrix(fpIn, &nn->biasesGradients[i])) {
            freeNeuralNetwork(nn);
            return NN_ERROR_IO_FILE;
        }
        if (!loadMatrix(fpIn, &nn->outputs[i])) {
            freeNeuralNetwork(nn);
            return NN_ERROR_IO_FILE;
        }
        if (!loadMatrix(fpIn, &nn->activations[i])) {
            freeNeuralNetwork(nn);
            return NN_ERROR_IO_FILE;
        }
    }

    *nnP = nn;
    return NN_OK;
}

NNStatus predictNN(const NeuralNetwork *nn, Matrix input, Matrix* output) {
    if (nn == NN_invalidP) return NN_ERROR_INVALID_ARGUMENT;
    if (input.cols != nn->architecture[0]) return NN_ERROR_INVALID_ARGUMENT;

    if (!forward(*nn, input)) return NN_ERROR_MEMORY_ALLOCATION;
    *output = copyMatrix(nn->activations[nn->layerCount - 1]);
    if (output->elements == MATRIX_invalidP) return NN_ERROR_MEMORY_ALLOCATION;
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
            return sqrtf(2.0f / nIn);
            break;
        case NN_INITIALIZATION_HE_UNIFORM:
            return sqrtf(6.0f / nIn);
            break;
        case NN_INITIALIZATION_XAVIER:
            return sqrtf(2.0f / (nIn+nOut));
            break;
        case NN_INITIALIZATION_XAVIER_UNIFORM:
            return sqrtf(6.0f / (nIn + nOut));
            break;
        default: return sqrtf(2.0f / nIn); // default to HE
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

    float z0 = (float)(sqrtf(-2.0f * log(u1)) * cos(2.0f * M_PI * u2));
    
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
    if (result.elements == MATRIX_invalidP) return EMPTY_MATRIX ;

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
    // handle perfect predictions exactly to avoid floating point issues
    if ((expectedOutput == 1.0f && output == 1.0f) ||
        (expectedOutput == 0.0f && output == 0.0f)) {
        return 0.0f;
    }

    // clip output to avoid log(0)
    float clipped_output = fmaxf(NN_epsilon, fminf(1.0f - NN_epsilon, output));
    return -(expectedOutput * log(clipped_output) + (1 - expectedOutput) * log(1 - clipped_output));
}

float BCElossDerivative(float output, float expectedOutput) {
    return  output - expectedOutput; //simplified derivative
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

void createNNConfig(NNConfig *config) {
    config->learningRate = -1.0f;
    config->hiddenLayersAF = (NNActivationFunction) 1000;
    config->lossFunction = (NNLossFunction) 1000;
    config->outputLayerAF = (NNActivationFunction) 1000;
    config->weightInitializerF = (NNInitializationFunction) 1000;
}

NNStatus checkNNConfig(NNConfig config) {
    if (config.learningRate <= 0.0f)return NN_ERROR_INVALID_LEARNING_RATE;

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
            return "An error occured while storing / loading the NeuralNetwork.";
            break;
        case NN_ERROR_UNKNOWN:
            return "The error was not recognized";
            break;
        default:
            return "Unknown error";
            break;
    }
}
