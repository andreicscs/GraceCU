
#include <stdlib.h>
#include <time.h>
#include "NeuralNetwork.h"
#include "Matrix.h"
#include <math.h>

void forward(NeuralNetwork nn, Matrix input);
void backPropagation(NeuralNetwork nn, Matrix expectedOutput);
Matrix applyActivation(NeuralNetwork nn, Matrix matrix, int iLayer);
Matrix multipleOutputActivationFunction(Matrix mat, int af);
double activationFunction(double x, int af);
Matrix computeOutputLayerDeltas(NeuralNetwork nn, Matrix expectedOutput);
Matrix computeActivationDerivative(Matrix outputs, int af);
Matrix computeMultipleOutputLossDerivativeMatrix(Matrix output, Matrix expectedOutput, int lf);
double AFDerivative(double x, int af);
Matrix CCElossDerivativeMatrix(Matrix predicted, Matrix expected);
Matrix computeOutputLayerDeltas(NeuralNetwork nn, Matrix expectedOutput);
Matrix computeActivationDerivative(Matrix outputs, int af);
Matrix computeLossDerivative(Matrix outputs, Matrix expectedOutputs, int lf);
double lossDerivative(double output, double expectedOutput, int lf);
double BCEloss(double output, double expectedOutput);
double BCElossDerivative(double output, double expectedOutput);
double MSEloss(double output, double expectedOutput);
double MSElossDerivative(double output, double expectedOutput);
void initializeMatrixRand(Matrix mat);
Matrix softmax(Matrix mat);
double reluDerivative(double x);
double relu(double x);
double sigmoid(double x);
void updateWeightsAndBiases(NeuralNetwork nn, int batchSize);
double loss(double output, double expectedOutput, int lf);
double multipleOutputLoss(Matrix output, Matrix expectedOutput, int lf);
double CCEloss(Matrix predictions, Matrix labels);
double computeSingleOutputAccuracy(NeuralNetwork nn, Matrix dataset);
double computeMultiClassAccuracy(NeuralNetwork nn, Matrix dataset, int nOutputs);


NeuralNetwork createNeuralNetwork(int* architecture, int layerCount) {
	NeuralNetwork nn;
    
    nn.architecture = architecture;
    nn.layerCount = layerCount;
    nn.weights = (Matrix*) malloc(sizeof(Matrix)*layerCount); if (nn.weights == NULL)throw "createNeuralNetwork: malloc failed";
    nn.biases = (Matrix*)malloc(sizeof(Matrix) * layerCount); if (nn.biases == NULL)throw "createNeuralNetwork: malloc failed";
    nn.weightsGradients = (Matrix*)malloc(sizeof(Matrix) * layerCount); if (nn.weightsGradients == NULL)throw "createNeuralNetwork: malloc failed";
    nn.biasesGradients = (Matrix*)malloc(sizeof(Matrix) * layerCount); if (nn.biasesGradients == NULL)throw "createNeuralNetwork: malloc failed";
    nn.inputGradients = (Matrix*)malloc(sizeof(Matrix) * layerCount); if (nn.inputGradients == NULL)throw "createNeuralNetwork: malloc failed";
    nn.outputs = (Matrix*)malloc(sizeof(Matrix) * layerCount); if (nn.outputs == NULL)throw "createNeuralNetwork: malloc failed";
    nn.activations = (Matrix*)malloc(sizeof(Matrix) * layerCount); if (nn.activations == NULL)throw "createNeuralNetwork: malloc failed";
    nn.numOutputs = architecture[layerCount - 1];


    /*
    * allocating first layer matrices.
    */
    nn.weights[0] = createMatrix(1, architecture[0]);
    nn.biases[0] = createMatrix(1, architecture[0]);
    
    nn.weightsGradients[0] = createMatrix(1, architecture[0]);
    nn.biasesGradients[0] = createMatrix(1, architecture[0]);
    nn.inputGradients[0] = createMatrix(1, architecture[0]);
    
    nn.outputs[0] = createMatrix(1, architecture[0]);
    nn.activations[0] = createMatrix(1, architecture[0]);

    /*
    * initializing first layer matrices.
    * weights and biases are initialized to neutral values for the corresponding 
    * operations to make sure they are ignored for the input layer.
    * other matrices are initialized to 0.
    */
    fillMatrix(nn.weights[0], 1);
    fillMatrix(nn.biases[0], 0);
    fillMatrix(nn.weightsGradients[0], 0);
    fillMatrix(nn.biasesGradients[0], 0);
    fillMatrix(nn.inputGradients[0], 0);
    fillMatrix(nn.outputs[0], 0);
    fillMatrix(nn.activations[0], 0);

    srand((unsigned int)time(NULL));   // Initialization, should only be called once.

    // !!TO DO
    //should implement different initialization methods for different activation functions, or maybe let the user decide which initialization method to use, 
    //initializing weights in the range (-0.5, 0.5) and biases to 0.01(which helps prevent dead neurons) for now.
    for (int i = 1; i < layerCount; i++) {
        nn.weights[i] = createMatrix(architecture[i-1], architecture[i]);
        nn.biases[i] = createMatrix(1, architecture[i]);
        nn.weightsGradients[i] = createMatrix(architecture[i-1], architecture[i]);
        nn.biasesGradients[i] = createMatrix(1, architecture[i]);
        nn.inputGradients[i] = createMatrix(architecture[i-1], architecture[i]);
        nn.outputs[i] = createMatrix(1, architecture[i]);
        nn.activations[i] = createMatrix(1, architecture[i]);
    
        initializeMatrixRand(nn.weights[i]);
        fillMatrix(nn.biases[i], 0.01);
        fillMatrix(nn.weightsGradients[i], 0);
        fillMatrix(nn.biasesGradients[i], 0);
        fillMatrix(nn.inputGradients[i], 0);
        fillMatrix(nn.outputs[i], 0);
        fillMatrix(nn.activations[i], 0);
    }
    nn.learningRate = 0.1;

	return nn;
}




void freeNeuralNetwork(NeuralNetwork nn) {
    //free all matrices inside matrices arrays
    for (int i = 0; i < nn.layerCount; i++) {
        freeMatrix(nn.weights[i]);
        freeMatrix(nn.biases[i]);
        freeMatrix(nn.weightsGradients[i]);
        freeMatrix(nn.biasesGradients[i]);
        freeMatrix(nn.inputGradients[i]);
        freeMatrix(nn.outputs[i]);
        freeMatrix(nn.activations[i]);
    }

    //free matrices arrays
    free(nn.weights);
    free(nn.biases);
    free(nn.weightsGradients);
    free(nn.biasesGradients);
    free(nn.inputGradients);
    free(nn.outputs);
    free(nn.activations);


    //reset values
    nn.architecture = 0;
    nn.layerCount = 0;
    nn.numOutputs = 0;
    nn.learningRate = 0;
}




// !! TO DO look into SGD implementation
void trainNN(NeuralNetwork nn, Matrix trainingData, int batchSize) {
    int trainCount = trainingData.rows;

    // loop over training data
    for (int i = 0; i < trainCount; ++i) {
        forward(nn, getSubMatrix(trainingData, i, 0, 1, trainingData.cols - nn.numOutputs));
        backPropagation(nn, getSubMatrix(trainingData, i, trainingData.cols - nn.numOutputs, 1, nn.numOutputs));
        if ((i + 1) % batchSize == 0 || i == trainingData.rows - 1) { // udate weights only after batchSize rows are processed.
            updateWeightsAndBiases(nn, batchSize);
        }
    }
}


void forward(NeuralNetwork nn, Matrix input) {
    nn.activations[0] = input;
    nn.outputs[0] = nn.activations[0];
    for (int i = 1; i < nn.layerCount; ++i) {
        nn.activations[i] = multiplyMatrix(nn.activations[i - 1], nn.weights[i]);
        addMatrixInPlace(nn.activations[i], nn.biases[i]);
        nn.outputs[i] = nn.activations[i];
        nn.activations[i] = applyActivation(nn, nn.activations[i], i);
    }
}

void backPropagation(NeuralNetwork nn, Matrix expectedOutput) {
    Matrix* deltas = (Matrix*)malloc(sizeof(Matrix) * nn.layerCount); if (deltas == NULL)throw "backPropagation: malloc failed";
    deltas[nn.layerCount - 1] = computeOutputLayerDeltas(nn, expectedOutput);

    // propagate error backward
    for (int i = nn.layerCount - 1; i > 0; --i) {
        Matrix weightGradients = multiplyMatrix(transposeMatrix(nn.activations[i - 1]), deltas[i]);  // compute weight gradients for layer i Weight gradients = activations[i-1]^T * deltas[i]
        addMatrixInPlace(nn.weightsGradients[i], weightGradients);

        // compute bias gradients (sum over batch)
        Matrix biasGradients = deltas[i];
        addMatrixInPlace(nn.biasesGradients[i], biasGradients);

        // compute deltas for previous layer (if not input layer)
        if (i > 1) {
            Matrix inputGradients = multiplyMatrix(deltas[i], transposeMatrix(nn.weights[i]));	// input gradients = deltas[i] * weights[i]^T
            Matrix activationDerivative = computeActivationDerivative(nn.outputs[i - 1], nn.hiddenLayersAF);	// activation derivative = f'(outputs[i-1])
            deltas[i - 1] = multiplyMatrixElementWise(inputGradients, activationDerivative); 	// deltas for previous layer = inputGradients ⊙ activationDerivative
        }
    }
}

void updateWeightsAndBiases(NeuralNetwork nn, int batchSize) {
    // this loop can be parallelized
    for (int i = 1; i < nn.layerCount; ++i) {
        // Update weights
        scaleMatrixInPlace(nn.weightsGradients[i], 1.0 / batchSize);
        subtractMatrixInPlace(nn.weights[i], scaleMatrix(nn.weightsGradients[i], nn.learningRate));
        fillMatrix(nn.weightsGradients[i], 0.0); // Reset gradients

        // Update biases
        scaleMatrixInPlace(nn.biasesGradients[i], 1.0 / batchSize);
        subtractMatrixInPlace(nn.biases[i], scaleMatrix(nn.biasesGradients[i], nn.learningRate));
        fillMatrix(nn.biasesGradients[i], 0.0); // Reset gradients
    }
}




Matrix computeOutputLayerDeltas(NeuralNetwork nn, Matrix expectedOutput) {
    Matrix predicted = nn.activations[nn.layerCount - 1];
    Matrix rawPredicted = nn.outputs[nn.layerCount - 1];
    Matrix curLayerDeltas;
    if (nn.numOutputs > 1) { // Multi-output
        curLayerDeltas = computeMultipleOutputLossDerivativeMatrix(predicted, expectedOutput, nn.lossFunction); // because the af derivative and the loss derivative simplify each other only one calculation is needed
    }
    else { // Single output
        Matrix dLoss_dY = computeLossDerivative(predicted, expectedOutput, nn.lossFunction); // derivative of the loss function
        Matrix activationDerivative = computeActivationDerivative(rawPredicted, nn.outputLayerAF);
        curLayerDeltas = multiplyMatrixElementWise(dLoss_dY, activationDerivative); // delta = dLoss_dY * derivative of the activation function with the non-activated output as input
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

Matrix computeActivationDerivative(Matrix outputs, int af) {
    Matrix derivative = createMatrix(outputs.rows, outputs.cols);
    for (int i = 0; i < outputs.rows; i++) {
        for (int j = 0; j < outputs.cols; j++) {
            derivative.elements[i * derivative.rows + j] = AFDerivative(outputs.elements[i * outputs.rows + j], af);
        }
    }
    return derivative;
}

Matrix computeLossDerivative(Matrix outputs, Matrix expectedOutputs, int lf) {
    Matrix derivative = createMatrix(outputs.rows, outputs.cols);
    for (int j = 0; j < outputs.cols; j++) {
        derivative.elements[j] = lossDerivative(outputs.elements[j], expectedOutputs.elements[j], lf);
    }
    return derivative;
}


double lossDerivative(double output, double expectedOutput, int lf) {
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


Matrix applyActivation(NeuralNetwork nn, Matrix mat, int iLayer) {
    Matrix activated;

    if ((iLayer == nn.layerCount - 1) && (nn.numOutputs > 1)) { // try to apply non mutually exclusive multiple clases AFs first.
        activated = multipleOutputActivationFunction(mat, nn.outputLayerAF);
        if (activated.elements != NULL) {
            return activated;
        }
    }
    activated = createMatrix(mat.rows, mat.cols);
    // if they were not selected proceed with mutually exclusive AF.
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            if (iLayer == nn.layerCount - 1) {
                activated.elements[i * activated.cols + j] = activationFunction(mat.elements[i * mat.cols + j], nn.outputLayerAF);
            }
            else {
                activated.elements[i * activated.cols + j] = activationFunction(mat.elements[i * mat.cols + j], nn.hiddenLayersAF);
            }
        }
    }
    return activated;
}

Matrix multipleOutputActivationFunction(Matrix mat, int af) {
    Matrix res = createMatrix(0,0);
    res.elements = NULL;

    switch (af) {
        case NN_SOFTMAX:
            return softmax(mat);
        default:
            break;
    }
    return res;
}

double activationFunction(double x, int af) {
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

double AFDerivative(double x, int af) {
    switch (af) {
        case NN_SIGMOID:
        {
            double sig = sigmoid(x);
            return sig * (1.0 - sig);
        }
        case NN_RELU:
            return reluDerivative(x);
        default:
            break;
        }
    return 1;
}

double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}
double relu(double x) {
    return 0 >= x ? 0 : x; // max between 0 and x
}

double reluDerivative(double x) {
    return x <= 0 ? 0 : 1;
}

Matrix softmax(Matrix mat) {
    Matrix result = createMatrix(mat.rows, mat.cols);

    for (int i = 0; i < mat.rows; i++) {
        double max = mat.elements[i * mat.cols];
        for (int j = 1; j < mat.cols; j++) {
            if (mat.elements[i * mat.cols + j] > max) {
                max = mat.elements[i * mat.cols + j];
            }
        }
        double sum = 0.0;
        for (int j = 0; j < mat.cols; j++) {
            result.elements[i * mat.cols + j] = exp(mat.elements[i * mat.cols + j] - max);
            sum += result.elements[i * mat.cols + j];
        }
        for (int j = 0; j < mat.cols; j++) {
            result.elements[i * mat.cols + j] /= sum;
        }
    }
    return result;
}

Matrix CCElossDerivativeMatrix(Matrix predicted, Matrix expected) {
    // For softmax + categorical cross-entropy, delta = predicted - true label
    Matrix deltas = subtractMatrix(predicted, expected);
    return deltas;
}

double BCEloss(double output, double expectedOutput) {
    // Clip output to avoid log(0)
    double epsilon = 1e-9;  // Small value to prevent log(0)
    output = epsilon >= (1 - epsilon <= output ? 1 - epsilon : output) ? epsilon : (1 - epsilon <= output ? 1 - epsilon : output);
    return -(expectedOutput * log(output) + (1 - expectedOutput) * log(1 - output));
}
double BCElossDerivative(double output, double expectedOutput) {
    // Clip output to avoid division by zero
    double epsilon = 1e-9;
    output = epsilon >= (1 - epsilon <= output ? 1 - epsilon : output) ? epsilon : (1 - epsilon <= output ? 1 - epsilon : output);
    return (output - expectedOutput) / (output * (1 - output));
}

double MSEloss(double output, double expectedOutput) {
    double error = 0;
    error = (output - expectedOutput);
    error = error * error;
    return error;
}

double MSElossDerivative(double output, double expectedOutput) {
    double error = 0;
    error = output - expectedOutput;
    return error;
}




double computeAverageLossNN(NeuralNetwork nn, Matrix trainingData) {
    int numSamples = trainingData.rows;
    double totalLoss = 0.0;
    
    for (int i = 0; i < numSamples; i++) {
        // Split input and expected output
        Matrix input = getSubMatrix(trainingData, i, 0, 1, trainingData.cols - nn.numOutputs);
        Matrix expected = getSubMatrix(trainingData, i, trainingData.cols - nn.numOutputs, 1, nn.numOutputs);
        
        // Forward pass
        forward(nn, input);
        Matrix prediction = nn.activations[nn.layerCount - 1];
        
        if(nn.numOutputs>1) {
            totalLoss+=multipleOutputLoss(prediction, expected, nn.lossFunction);
        }else {
            // Calculate loss for this example
            for (int j = 0; j < nn.numOutputs; j++) {
                totalLoss+=loss(prediction.elements[j], expected.elements[j], nn.lossFunction);
            }
        }    
    }
    return totalLoss / numSamples;
}

double loss(double output, double expectedOutput, int lf) {
    switch(lf) {
    case NN_MSE:
        return MSEloss(output,expectedOutput);
    case NN_BCE:
        return BCEloss(output,expectedOutput);
    default:
        break;
    }
    return MSEloss(output,expectedOutput);
}

double multipleOutputLoss(Matrix output, Matrix expectedOutput, int lf) {
    switch(lf) {
    case NN_CCE:
        return CCEloss(output,expectedOutput);
    default:
        throw "multipleOutputLoss: Unsupported loss function: " + lf;
           
    }
}


double CCEloss(Matrix predictions, Matrix labels) {
    if (predictions.rows != labels.rows || predictions.cols != labels.cols) {
        throw "CCEloss: Predictions and labels must have the same dimensions.";
    }

    double loss = 0.0;
    for (int i = 0; i < predictions.rows; i++) {
        for (int j = 0; j < predictions.cols; j++) {
            double predicted = predictions.elements[i * predictions.cols + j];
            double expected = labels.elements[i * labels.cols + j];

            // Ensure predicted values are valid probabilities
            if (predicted < 0 || predicted > 1) {
                throw "CCEloss: Predictions must be probabilities (0 < p <= 1).";
            }

            // CCE formula: -sum(y * log(p))
            loss += expected * log(predicted + 1e-10); // Add epsilon to avoid log(0)
        }
    }

    // Average the loss over all samples
    return -loss / predictions.rows;
}

double computeAccuracyNN(NeuralNetwork nn, Matrix dataset, int nOutputs) {
    if(nOutputs>1) {
        return computeMultiClassAccuracy(nn, dataset, nOutputs);
    }else {
        return computeSingleOutputAccuracy(nn, dataset);
    }
}

double computeMultiClassAccuracy(NeuralNetwork nn, Matrix dataset, int nOutputs) {
    int correct = 0;
    for(int i=0; i<dataset.rows; i++) {
        Matrix input = getSubMatrix(dataset, i, 0, 1, dataset.cols - nOutputs);
        Matrix output = getSubMatrix(dataset, i, dataset.cols - nOutputs, 1, nOutputs);
        
        forward(nn, input);
        Matrix pred = nn.activations[nn.layerCount-1];
        
        int predClass = 0;
        double maxVal = pred.elements[0];
        for(int j=1; j<nOutputs; j++) {
            if(pred.elements[j] > maxVal) {
                maxVal = pred.elements[j];
                predClass = j;
            }
        }
        
        int trueClass = 0;
        for(int j=0; j<nOutputs; j++) {
            if(output.elements[j] == 1.0) {
                trueClass = j;
                break;
            }
        }
        
        if(predClass == trueClass) correct++;
    }
    return (double)correct/dataset.rows*100;
}

double computeSingleOutputAccuracy(NeuralNetwork nn, Matrix dataset) {
    int numSamples = dataset.rows;
    int correct = 0;
    for (int i = 0; i < numSamples; i++) {
        Matrix input = getSubMatrix(dataset, i, 0, 1, dataset.cols - 1);
        Matrix expected = getSubMatrix(dataset, i, dataset.cols - 1, 1, 1);
        forward(nn, input);
        double prediction = nn.activations[nn.layerCount - 1].elements[0];
        int predictedLabel = (prediction >= 0.5) ? 1 : 0;
        int trueLabel = (int) expected.elements[0];
        if (predictedLabel == trueLabel) {
            correct++;
        }
    }
    return (double) correct / numSamples * 100; // Accuracy in percentage
}


void saveStateNN(NeuralNetwork nn){
    // !! TO DO implement saveStateNN
}


void initializeMatrixRand(Matrix mat) {
    int randomValue;
    for (int i = 0; i < mat.rows; i++) {
        for (int j = 0; j < mat.cols; j++) {
            randomValue = ((rand() % (1000+ 1))-500)/1000; // random value in range (-0.5, 0.5)
            mat.elements[i * mat.cols + j] = randomValue;
        }
    }
}







