
#include <stdlib.h>
#include <time.h>
#include "NeuralNetwork.h"
#include "Matrix.h"



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

    srand(time(NULL));   // Initialization, should only be called once.

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
    nn.hiddenLayersAF = "";
    nn.outputLayerAF = "";
    nn.lossFunction = "";

    

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
    nn.hiddenLayersAF = "";
    nn.outputLayerAF = "";
    nn.lossFunction = "";

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