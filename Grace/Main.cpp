 #include <stdio.h>
 #include "NeuralNetwork.h"

 void main(){
    int architecture[] = {2,4,1};
    NeuralNetwork nn = createNeuralNetwork(architecture, 3);
    
    Matrix trainingData = createMatrix(4,3);
    double data[] = {0,0,0,0,1,1,1,0,1,1,1,0};
    trainingData.elements;
    
    
    
    nn.learningRate = 0.01;
    nn.hiddenLayersAF = NN_RELU;
    nn.outputLayerAF = NN_SIGMOID;
    nn.lossFunction = NN_MSE;



    int epochs=10;
    for(int i=0; i<epochs; ++i){
        train(nn, trainingData, 4);

    }

 }