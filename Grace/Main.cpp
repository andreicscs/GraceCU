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

    // load and preprocess training data
    Matrix rawTrainData = loadMNIST(trainFilePath, numTrainSamples);
    Matrix normalizedTrainData = normalizeData(rawTrainData);
    Matrix trainLabels = oneHotEncodeLabels(normalizedTrainData);
    Matrix trainDataset = prepareDataset(normalizedTrainData, trainLabels);
    freeMatrix(normalizedTrainData);
    freeMatrix(trainLabels);
    

    
    // load and preprocess test data
    Matrix rawTestData = loadMNIST(testFilePath, numTestSamples);
    Matrix normalizedTestData = normalizeData(rawTestData);
    Matrix testLabels = oneHotEncodeLabels(normalizedTestData);
    Matrix testDataset = prepareDataset(normalizedTestData, testLabels);
    freeMatrix(normalizedTestData);
    freeMatrix(testLabels);
    

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
    float loss;
    float accuracy;
    for(int i=0; i<epochs; ++i){
        printf("Epoch %d\n",i+1);
        
        clock_t begin = clock();

        err=trainNN(nn, trainDataset, batchSize);
        if (err != 0) {
            printf("trainNN: %s\n", NNStatusToString(err));
            return -1;
        }
        computeAverageLossNN(nn, trainDataset, &loss);
        computeAccuracyNN(nn, trainDataset, &accuracy);
        printf("Training Loss: %f, Accuracy: %f\n", loss, accuracy);

        computeAverageLossNN(nn, testDataset, &loss);
        computeAccuracyNN(nn, testDataset, &accuracy);
        printf("Test Loss: %f, Accuracy: %f\n", loss, accuracy);
        
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
    freeMatrix(testDataset);

    
    _CrtDumpMemoryLeaks();

    return 0;
}



