#ifndef MATRIX_H
#define MATRIX_H

#define MATRIX_invalidP NULL
#include <stdio.h>

// !!TODO look into ways to improve information hiding
typedef struct {
	unsigned int rows;
	unsigned int cols;
	float* elements; // 1d array containing matrix data
} Matrix;

Matrix createMatrix(unsigned int rows, unsigned int cols);
void freeMatrix(Matrix mat);
Matrix multiplyMatrix(Matrix a, Matrix b);
Matrix multiplyMatrixElementWise(Matrix a, Matrix b);
bool scaleMatrixInPlace(Matrix mat, float scalar);
Matrix scaleMatrix(Matrix mat, float scalar);
bool addMatrixInPlace(Matrix a, Matrix b);
Matrix addMatrix(Matrix a, Matrix b); 
bool subtractMatrixInPlace(Matrix a, Matrix b);
Matrix subtractMatrix(Matrix a, Matrix b);
bool fillMatrix(Matrix mat, float value);
Matrix getSubMatrix(Matrix mat, unsigned int startRow, unsigned int startCol, unsigned int numRows, unsigned int numCols);
Matrix transposeMatrix(Matrix mat);
void printMatrix(Matrix mat);
Matrix copyMatrix(Matrix src);
float sumMatrix(Matrix src);
bool storeMatrix(Matrix mat, FILE *fpOut);
bool loadMatrix(FILE *fpIn, Matrix *out);

#endif // MATRIX_H