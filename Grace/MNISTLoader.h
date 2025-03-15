#ifndef MNIST_H
#define MNIST_H
#include "Matrix.h"


Matrix loadMNIST(const char* filePath, int numSamples);
Matrix normalizeData(Matrix data);
Matrix oneHotEncodeLabels(Matrix data);
Matrix prepareDataset(Matrix data, Matrix labels);

#endif // MNISTLoader.h