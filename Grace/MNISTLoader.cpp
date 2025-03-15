#include "MNISTLoader.h"
#include <iostream>
#include <fstream>
#include <sstream>

Matrix loadMNIST(const char* filePath, int numSamples) {
    Matrix dataset = createMatrix(numSamples, 785); // 784 pixels + 1 label

    std::ifstream file(filePath);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filePath << std::endl;
        return dataset;
    }

    std::string line;
    int row = 0;
    bool isHeader = false; // Flag to skip the header row

    while (std::getline(file, line) && row < numSamples) {
        if (isHeader) {
            isHeader = false; // Skip the header row
            continue;
        }

        std::stringstream ss(line);
        std::string value;
        int col = 0;

        while (std::getline(ss, value, ',') && col < 785) {
            dataset.elements[row * dataset.cols + col] = std::stod(value);
            col++;
        }
        row++;
    }

    file.close();
    return dataset;
}


Matrix normalizeData(Matrix data) {
    // Normalize pixel values to [0, 1]
    for (int i = 0; i < data.rows; i++) {
        for (int j = 1; j < 785; j++) { // Skip the label column (column 0)
            data.elements[i*data.cols+j] /= 255.0;
        }
    }
    return data;
}
Matrix oneHotEncodeLabels(Matrix data) {
    // Convert labels to one-hot encoded format
    Matrix labels = createMatrix(data.rows, 10);
    fillMatrix(labels, 0.0);  // Inizializza tutte le etichette a 0.0
    for (int i = 0; i < data.rows; i++) {
        int label = static_cast<int>(data.elements[i*data.cols]);
        labels.elements[i * labels.cols+label] = 1.0;
    }
    return labels;
}
Matrix prepareDataset(Matrix data, Matrix labels) {
    // Combine inputs (pixel values) and one-hot encoded labels into a single matrix
    Matrix dataset = createMatrix(data.rows, 794); // 784 inputs + 10 outputs
    for (int i = 0; i < data.rows; i++) {
        // Copy pixel values (columns 1-784 of data)
        for (int j = 1; j < 785; j++) {
            dataset.elements[i*dataset.cols+j - 1] = data.elements[i*data.cols+j];
        }
        // Copy one-hot encoded labels (columns 785-794)
        for (int j = 0; j < 10; j++) {
            dataset.elements[i*dataset.cols+ 784 + j] = labels.elements[i*labels.cols+j];
        }
    }
    return dataset;
}