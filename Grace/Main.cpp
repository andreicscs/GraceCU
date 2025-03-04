
// check memory leaks
#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>

#ifdef _DEBUG
#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new DEBUG_NEW
#define malloc(s) _malloc_dbg(s, _NORMAL_BLOCK, __FILE__, __LINE__)
#endif




#include <stdio.h>
#include "NeuralNetwork.h"
#include "MNISTLoader.h"
#include <iostream>
#include <chrono>;



using namespace std;

 void main(){

    
    // Load and preprocess data
    char trainFilePath [] = "C:\\Users\\termi\\Desktop\\mnist_train.csv";
    char testFilePath [] = "C:\\Users\\termi\\Desktop\\mnist_test.csv";

    int numTrainSamples = 60000; // MNIST training set size
    int numTestSamples = 10000;  // MNIST test set size

    // Load and preprocess training data
    Matrix trainData = loadMNIST(trainFilePath, numTrainSamples);
    trainData = normalizeData(trainData);
    Matrix trainLabels = oneHotEncodeLabels(trainData);

    Matrix trainDataset = prepareDataset(trainData, trainLabels);

    
    // Load and preprocess test data
    Matrix testData = loadMNIST(testFilePath, numTestSamples);
    testData = normalizeData(testData);
    Matrix testLabels = oneHotEncodeLabels(testData);
    Matrix testDataset = prepareDataset(testData, testLabels);
    
    

    int architecture[] = { 784, 128, 64, 10 };

    NeuralNetwork nn = createNeuralNetwork(architecture, 4);


    nn.learningRate = 0.1;
    nn.hiddenLayersAF = NN_RELU;
    nn.outputLayerAF = NN_SOFTMAX;
    nn.lossFunction = NN_CCE;
    nn.numOutputs = 10;


    int batchSize = 32;
    int epochs=100;
    
     
    /*
    int architecture[] = { 2,4,1 };
    NeuralNetwork nn = createNeuralNetwork(architecture, 3);

     
    Matrix trainDataset = createMatrix(4, 3);
    float data[] = { 0,0,0,
                    0,1,1,
                    1,0,1,
                    1,1,0 };

    trainDataset.elements = data;


    nn.learningRate = 0.5;
    nn.hiddenLayersAF = NN_RELU;
    nn.outputLayerAF = NN_SIGMOID;
    nn.lossFunction = NN_BCE;
    nn.numOutputs = 1;


    int batchSize = 4;
    int epochs = 1000;
    */
    for(int i=0; i<epochs; ++i){
        cout << "Epoch " << (i + 1)<< endl;
        auto begin = chrono::high_resolution_clock::now();

        trainNN(nn, trainDataset, batchSize);
        float trainLoss = computeAverageLossNN(nn, trainDataset);
        float trainAccuracy = computeAccuracyNN(nn, trainDataset);
        cout << "Training Loss : " << trainLoss << ", Accuracy : " << trainAccuracy << endl;

        float testLoss = computeAverageLossNN(nn, testDataset);
        float testAccuracy = computeAccuracyNN(nn, testDataset);
        cout << "Test Loss : " << testLoss << ", Accuracy : " << testAccuracy << endl;

        auto end = chrono::high_resolution_clock::now();
        auto dur = end - begin;
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();

        cout << "Epoch time: " << ms/1000 << " seconds"<< endl;
    }

    freeNeuralNetwork(nn);
    freeMatrix(trainDataset);
 }



