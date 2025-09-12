#include "NeuralNetworkDebug.h"
#include "Matrix.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

// small epsilon for floating point comparison
#define EPS 1e-5
#define fequal(a, b) (fabs((a) - (b)) < EPS)


// Internal activation function tests

void testRelu() {
    printf("Testing relu...\n");
    assert(fequal(relu(5.0f), 5.0f));
    assert(fequal(relu(-3.0f), 0.0f));
}

void testReluDerivative() {
    printf("Testing reluDerivative...\n");
    assert(fequal(reluDerivative(5.0f), 1.0f));
    assert(fequal(reluDerivative(-3.0f), 0.0f));
}

void testSigmoid() {
    printf("Testing sigmoid...\n");
    assert(fequal(sigmoid(0.0f), 0.5f));
    assert(sigmoid(10.0f) > 0.999f);
    assert(sigmoid(-10.0f) < 0.001f);
}

void testSigmoidDerivative() {
    printf("Testing sigmoidDerivative...\n");
    float s = sigmoid(0.0f);
    assert(fequal(sigmoidDerivative(s), s * (1 - s)));
}

void testSoftmax() {
    printf("Testing softmax...\n");
    float data[3] = { 1.0f, 2.0f, 3.0f };
    Matrix m = createMatrix(3, 1);
    for (int i = 0; i < 3; i++) m.elements[i] = data[i];
    Matrix sm = softmax(m);

    float sum = 0.0f;
    for (int i = 0; i < 3; i++) sum += sm.elements[i];
    assert(fequal(sum, 1.0f));

    freeMatrix(m);
    freeMatrix(sm);
}


// Internal loss function tests

void testCceLoss() {
    printf("Testing CCEloss...\n");

    // perfect prediction
    Matrix pred1 = createMatrix(1, 3);
    Matrix label1 = createMatrix(1, 3);
    float data1[] = { 0.0f, 1.0f, 0.0f };
    float data2[] = { 0.0f, 1.0f, 0.0f };
    memcpy(pred1.elements, data1, sizeof(data1));
    memcpy(label1.elements, data2, sizeof(data2));

    assert(fequal(CCEloss(pred1, label1), 0.0f));

    // wrong prediction
    Matrix pred2 = createMatrix(1, 3);
    Matrix label2 = createMatrix(1, 3);
    float data3[] = { 1.0f, 0.0f, 0.0f };
    float data4[] = { 0.0f, 1.0f, 0.0f };
    memcpy(pred2.elements, data3, sizeof(data3));
    memcpy(label2.elements, data4, sizeof(data4));

    assert(CCEloss(pred2, label2) > 0.0f);

    freeMatrix(pred1);
    freeMatrix(label1);
    freeMatrix(pred2);
    freeMatrix(label2);
}

void testCceLossDerivative() {
    printf("Testing CCElossDerivativeMatrix...\n");

    Matrix predicted = createMatrix(1, 3);
    Matrix expected = createMatrix(1, 3);
    float data1[] = { 0.1f, 0.7f, 0.2f };
    float data2[] = { 0.0f, 1.0f, 0.0f };
    memcpy(predicted.elements, data1, sizeof(data1));
    memcpy(expected.elements, data2, sizeof(data2));

    Matrix derivative = CCElossDerivativeMatrix(predicted, expected);

    assert(fequal(derivative.elements[0], 0.1f));
    assert(fequal(derivative.elements[1], -0.3f));
    assert(fequal(derivative.elements[2], 0.2f));

    freeMatrix(predicted);
    freeMatrix(expected);
    freeMatrix(derivative);
}

void testBceLoss() {
    printf("Testing BCEloss...\n");
    assert(fequal(BCEloss(1.0f, 1.0f), 0.0f));  // perfect prediction
    assert(BCEloss(0.5f, 1.0f) > 0.0f);         // some error
}

void testBceLossDerivative() {
    printf("Testing BCElossDerivative...\n");
    // for y=1, derivative = -1/x
    assert(fabs(BCElossDerivative(1.0f, 1.0f) - -1.0f) < EPS);
}

void testMseLoss() {
    printf("Testing MSEloss...\n");
    assert(fequal(MSEloss(1.0f, 1.0f), 0.0f));
    assert(fequal(MSEloss(0.0f, 1.0f), 0.5f)); // (0-1)^2 / 2
}

void testMseLossDerivative() {
    printf("Testing MSElossDerivative...\n");
    assert(fequal(MSElossDerivative(1.0f, 1.0f), 0.0f));
    assert(fequal(MSElossDerivative(0.0f, 1.0f), -1.0f));
}


// Initialization and random helper tests

void testInitializeMatrixRand() {
    printf("Testing initializeMatrixRand...\n");

    // basic functionality with mean 0, stddev 1
    Matrix m1 = createMatrix(10, 10);
    initializeMatrixRand(m1, 0.0f, 1.0f);

    // verify that all elements are initialized (not zero)
    int nonZeroCount = 0;
    for (unsigned int i = 0; i < m1.rows * m1.cols; i++) {
        if (fabs(m1.elements[i]) > 1e-10f) {
            nonZeroCount++;
        }
    }
    assert(nonZeroCount > 0);

    // different mean and stddev
    Matrix m2 = createMatrix(5, 5);
    initializeMatrixRand(m2, 5.0f, 2.0f);

    // calculate sample mean and stddev
    float sum = 0.0f;
    float sumSq = 0.0f;
    int count = m2.rows * m2.cols;

    for (unsigned int i = 0; i < count; i++) {
        sum += m2.elements[i];
        sumSq += m2.elements[i] * m2.elements[i];
    }

    float sampleMean = sum / count;
    float sampleVariance = (sumSq / count) - (sampleMean * sampleMean);
    float sampleStddev = sqrtf(sampleVariance);

    // verify mean and stddev are approximately correct
    assert(fabs(sampleMean - 5.0f) < 1.0f);
    assert(fabs(sampleStddev - 2.0f) < 1.0f);

    // zero stddev (all values should equal mean)
    Matrix m3 = createMatrix(3, 3);
    initializeMatrixRand(m3, 3.0f, 0.0f);

    for (unsigned int i = 0; i < m3.rows * m3.cols; i++) {
        assert(fequal(m3.elements[i], 3.0f));
    }

    // negative mean
    Matrix m4 = createMatrix(4, 4);
    initializeMatrixRand(m4, -2.0f, 1.0f);

    int negativeCount = 0;
    for (unsigned int i = 0; i < m4.rows * m4.cols; i++) {
        if (m4.elements[i] < 0.0f) {
            negativeCount++;
        }
    }
    assert(negativeCount > 0);

    freeMatrix(m1);
    freeMatrix(m2);
    freeMatrix(m3);
    freeMatrix(m4);
}

void testRandomNormal() {
    printf("Testing randomNormal...\n");

    // basic mean and stddev
    const int samples = 10000;
    float mean = 5.0f;
    float stddev = 2.0f;

    float sum = 0.0f;
    float sumSq = 0.0f;

    for (int i = 0; i < samples; i++) {
        float r = randomNormal(mean, stddev);
        sum += r;
        sumSq += r * r;
    }

    float sampleMean = sum / samples;
    float sampleVariance = (sumSq / samples) - (sampleMean * sampleMean);
    float sampleStddev = sqrtf(sampleVariance);

    // verify mean and stddev are approximately correct
    assert(fabs(sampleMean - mean) < 0.2f);
    assert(fabs(sampleStddev - stddev) < 0.2f);

    // zero stddev should always return mean
    for (int i = 0; i < 100; i++) {
        float r = randomNormal(mean, 0.0f);
        assert(fequal(r, mean));
    }

    // different mean (negative)
    for (int i = 0; i < 100; i++) {
        float r = randomNormal(-3.0f, 1.0f);
        if (r < 0.0f) {
            return; // success if at least one value < 0
        }
    }
}

void testInitializationFunction() {
    printf("Testing initializationFunction...\n");

    unsigned int nIn = 10, nOut = 5;

    assert(fequal(initializationFunction(NN_INITIALIZATION_0, nIn, nOut), 0.0f));
    assert(fequal(initializationFunction(NN_INITIALIZATION_1, nIn, nOut), 1.0f));
    assert(fequal(initializationFunction(NN_INITIALIZATION_SPARSE, nIn, nOut), NN_epsilon));
    float he = initializationFunction(NN_INITIALIZATION_HE, nIn, nOut);
    assert(fequal(he, sqrtf(2.0f / nIn)));
    float heu = initializationFunction(NN_INITIALIZATION_HE_UNIFORM, nIn, nOut);
    assert(fequal(heu, sqrtf(6.0f / nIn)));
    float xavier = initializationFunction(NN_INITIALIZATION_XAVIER, nIn, nOut);
    assert(fequal(xavier, sqrtf(2.0f / (nIn + nOut))));
    float xavieru = initializationFunction(NN_INITIALIZATION_XAVIER_UNIFORM, nIn, nOut);
    assert(fequal(xavieru, sqrtf(6.0f / (nIn + nOut))));
    float def = initializationFunction((NNInitializationFunction)999, nIn, nOut);
    assert(fequal(def, sqrtf(2.0f / nIn)));
}




int main() {
    srand((unsigned int)time(NULL));

    testSigmoid();
    testSigmoidDerivative();
    testRelu();
    testReluDerivative();
    testMseLoss();
    testMseLossDerivative();
    testBceLoss();
    testBceLossDerivative();
    testSoftmax();
    testCceLoss();
    testCceLossDerivative();
    testInitializeMatrixRand();
    testRandomNormal();
    testInitializationFunction();

    printf("\nAll math tests passed!\n");
    return 0;
}
