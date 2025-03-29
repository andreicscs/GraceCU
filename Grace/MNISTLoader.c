#include "Matrix.h" // Include your matrix library header
#include <stdio.h>   // For FILE, fopen, fgets, fclose
#include <stdlib.h>  // For strtod, malloc, calloc, free
#include <string.h>  // For pointer manipulation (implicit)
#include <stdbool.h> // For bool type (C99 or later)
#include <errno.h>   // For strtod error checking (optional but good)

#define MAX_LINE_LENGTH 8192 // Reasonable buffer size for a CSV line

/*
 * Loads MNIST dataset from a CSV file.
 * Assumes CSV format: label,pixel1,pixel2,...,pixel784
 * Skips the first (header) line.
 * Returns a Matrix (numSamples x 785). On error, returns an invalid Matrix.
 */
Matrix loadMNIST(const char* filePath, int numSamples) {
    Matrix dataset = createMatrix(numSamples, 785); // 784 pixels + 1 label
    if (dataset.elements == NULL) {
        return dataset; // Allocation failed
    }

    FILE* file = fopen(filePath, "r");
    if (file == NULL) {
        freeMatrix(dataset); // Free allocated memory before returning
        dataset.rows = 0;
        dataset.cols = 0;
        dataset.elements = NULL;
        return dataset; // File open error
    }

    char line[MAX_LINE_LENGTH];
    int row = 0;

    // Skip header line
    if (fgets(line, sizeof(line), file) == NULL) {
        fclose(file);
        freeMatrix(dataset);
        dataset.rows = 0;
        dataset.cols = 0;
        dataset.elements = NULL;
        return dataset; // File empty or read error
    }

    // Read sample data
    while (row < numSamples && fgets(line, sizeof(line), file) != NULL) {
        char* ptr = line;
        char* endptr;
        int col = 0;

        while (col < 785 && *ptr != '\0' && *ptr != '\n') {
            errno = 0; // Reset errno for strtod
            float value = strtof(ptr, &endptr);

            // Basic check: did strtod consume any characters?
            if (ptr == endptr || errno != 0) {
                // Handle parsing error silently: maybe use 0.0 or break row processing
                value = 0.0;
                // Attempt to find next comma or end of line to potentially recover
                while (*endptr != ',' && *endptr != '\0' && *endptr != '\n') {
                    if (*endptr == '\0' || *endptr == '\n') break; // Stop if EOL/EOF hit
                    endptr++;
                }
            }

            dataset.elements[row * dataset.cols + col] = value;
            col++;

            ptr = endptr;
            if (*ptr == ',') {
                ptr++;
            }
            // Skip potential leading whitespace for next number
            while (*ptr == ' ' || *ptr == '\t') {
                ptr++;
            }
        }
        // Optional: could check if col == 785 here, but skipping warnings per request
        row++;
    }

    fclose(file);

    // If fewer samples were read than requested, adjust matrix rows number
    if (row < numSamples) {
        // Caller should ideally check matrix.rows, but we update it here
        dataset.rows = row;
        // Note: Memory is still allocated for numSamples rows.
        // Reallocation could be done but adds complexity.
    }

    return dataset;
}

/*
 * Normalizes pixel values (columns 1-784) to the [0, 1] range.
 * Modifies the input matrix 'data' in-place.
 */
Matrix normalizeData(Matrix data) {
    if (data.elements == NULL || data.cols < 785) {
        // Return original if invalid or not enough columns
        return data;
    }

    for (unsigned int i = 0; i < data.rows; i++) {
        // Skip the label column (index 0)
        for (int j = 1; j < 785; j++) {
            data.elements[i * data.cols + j] /= 255.0;
        }
    }
    return data; // Return the modified matrix
}

/*
 * Converts labels (column 0 of 'data') to one-hot encoding.
 * Returns a new Matrix (data.rows x 10).
 * Returns an invalid Matrix on error.
 */
Matrix oneHotEncodeLabels(Matrix data) {
    if (data.elements == NULL || data.rows == 0 || data.cols == 0) {
        return createMatrix(0, 0); // Return invalid matrix
    }

    Matrix labels = createMatrix(data.rows, 10);
    if (labels.elements == NULL) {
        return labels; // Allocation failed
    }

    fillMatrix(labels, 0.0); // Initialize all elements to 0.0

    for (unsigned int i = 0; i < data.rows; i++) {
        // Label is in the first column
        int label = (int)(data.elements[i * data.cols + 0]);

        // Check if label is valid (0-9)
        if (label >= 0 && label < 10) {
            labels.elements[i * labels.cols + label] = 1.0;
        }
        // Else: Invalid label, row remains all zeros (silent handling)
    }
    return labels;
}

/*
 * Combines normalized pixel data (cols 1-784 of 'data')
 * and one-hot encoded 'labels' into a single dataset matrix.
 * Returns a new Matrix (data.rows x 794).
 * Returns an invalid Matrix on error or invalid input.
 */
Matrix prepareDataset(Matrix data, Matrix labels) {
    // Input validation
    if (data.elements == NULL || labels.elements == NULL ||
        data.rows != labels.rows || data.rows == 0 ||
        data.cols < 785 || labels.cols != 10)
    {
        return createMatrix(0, 0); // Return invalid matrix
    }

    int num_inputs = 784;
    int num_outputs = 10;
    Matrix dataset = createMatrix(data.rows, num_inputs + num_outputs); // 784 inputs + 10 outputs
    if (dataset.elements == NULL) {
        return dataset; // Allocation failed
    }

    for (unsigned int i = 0; i < data.rows; i++) {
        // Copy pixel values (columns 1-784 from data -> 0-783 in dataset)
        for (int j = 1; j < 785; j++) {
            dataset.elements[i * dataset.cols + (j - 1)] = data.elements[i * data.cols + j];
        }
        // Copy one-hot labels (columns 0-9 from labels -> 784-793 in dataset)
        for (int j = 0; j < num_outputs; j++) {
            dataset.elements[i * dataset.cols + num_inputs + j] = labels.elements[i * labels.cols + j];
        }
    }
    return dataset;
}
