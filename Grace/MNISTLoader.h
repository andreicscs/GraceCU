#ifndef MNIST_H
#define MNIST_H
#include "Matrix.h"
#include <string>


Matrix loadMNIST(const std::string& filePath, int numSamples);
Matrix normalizeData(Matrix data);
Matrix oneHotEncodeLabels(Matrix data);
Matrix prepareDataset(Matrix data, Matrix labels);

#endif // MNISTLoader.h