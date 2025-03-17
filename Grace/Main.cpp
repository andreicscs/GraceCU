// check memory leaks
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>

#ifdef _DEBUG
#define malloc(s) _malloc_dbg(s, _NORMAL_BLOCK, __FILE__, __LINE__)
#endif

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "NeuralNetwork.h"
#include "MNISTLoader.h"


 int main(){
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


    NNConfig config;
    config.learningRate = 0.1f;
    config.weightInitializerF = NN_INITIALIZATION_HE;
    config.hiddenLayersAF = NN_ACTIVATION_RELU;
    config.outputLayerAF = NN_ACTIVATION_SOFTMAX;
    config.lossFunction = NN_LOSS_CCE;

    NeuralNetwork* nn;
    
    NNStatus err;
    
    err=createNeuralNetwork(architecture, 4, config, &nn);
    if (err!=0) {
        printf("createNeuralNetowork: %s\n", NNStatusToString(err));
        return -1;
    }
   
    int batchSize = 32;
    int epochs=1;
    
    float trainLoss;
    float trainAccuracy;
    for(int i=0; i<epochs; ++i){
        printf("Epoch %d\n",i+1);
        
        clock_t begin = clock();

        err=trainNN(nn, trainDataset, batchSize);
        if (err != 0) {
            printf("trainNN: %s\n", NNStatusToString(err));
            return -1;
        }
        computeAverageLossNN(nn, trainDataset, &trainLoss);
        computeAccuracyNN(nn, trainDataset, &trainAccuracy);
        printf("Training Loss: %f, Accuracy: %f\n", trainLoss, trainAccuracy);

        computeAverageLossNN(nn, testDataset, &trainLoss);
        computeAccuracyNN(nn, testDataset, &trainAccuracy);
        printf("Test Loss: %f, Accuracy: %f\n", trainLoss, trainAccuracy);
        
        clock_t end = clock();
        float dur = (float)(end - begin) / CLOCKS_PER_SEC;
        
        printf("Epoch time: %f, seconds\n", dur);        
    }

    err=freeNeuralNetwork(nn);
    if (err != 0) {
        printf("freeNeuralNetwork: %s\n", NNStatusToString(err));
        return -1;
    }
    freeMatrix(trainDataset);
    _CrtDumpMemoryLeaks();

    return 0;
 }



