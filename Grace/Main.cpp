#include <stdio.h>
#include "NeuralNetwork.h"
#include <iostream>
#include <chrono>

using namespace std;
/*
void debugNNSize(NeuralNetwork nn) {
    for (int i = 0; i < nn.layerCount; i++) {
        cout << "Layer " << i << endl;
        cout << "Weights: " << nn.weights[i].rows << " rows x " << nn.weights[i].cols << " cols" << endl;
        cout << "Biases: " << nn.biases[i].rows << " rows x " << nn.biases[i].cols << " cols" << endl;
        cout << "Weights Gradients: " << nn.weightsGradients[i].rows << " rows x " << nn.weightsGradients[i].cols << " cols" << endl;
        cout << "Biases Gradients: " << nn.biasesGradients[i].rows << " rows x " << nn.biasesGradients[i].cols << " cols" << endl;
        cout << "Outputs: " << nn.outputs[i].rows << " rows x " << nn.outputs[i].cols << " cols" << endl;
        cout << "Activations: " << nn.activations[i].rows << " rows x " << nn.activations[i].cols << " cols" << endl;
        cout << endl;
    }
}
*/



 void main(){
     
    int architecture[] = {2,4,1};
    NeuralNetwork nn = createNeuralNetwork(architecture, 3);


    Matrix trainingData = createMatrix(4,3);
    double data[] = {0,0,0,
                     0,1,1,
                     1,0,1,
                     1,1,0};

    trainingData.elements = data;
    
    
    nn.learningRate = 0.5;
    nn.hiddenLayersAF = NN_RELU;
    nn.outputLayerAF = NN_SIGMOID;
    nn.lossFunction = NN_BCE;
    nn.numOutputs = 1;


    int batchSize = 4;
    int epochs=1000;

    for(int i=0; i<epochs; ++i){
        auto begin = chrono::high_resolution_clock::now();

        trainNN(nn, trainingData, batchSize);
        double trainLoss = computeAverageLossNN(nn, trainingData);
        double trainAccuracy = computeAccuracyNN(nn, trainingData);
        
        auto end = chrono::high_resolution_clock::now();
        auto dur = end - begin;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();

        cout << "Epoch " << (i + 1) << " Training Loss: " << trainLoss << ", Accuracy: " << trainAccuracy << "%" << " time: " << ms << endl;
    }
 }



