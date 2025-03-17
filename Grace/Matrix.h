#ifndef MATRIX_H
#define MATRIX_H

typedef enum {
	MATRIX_OK, // no errors
	MATRIX_ERROR_OUTOFBOUNDS, // matrix out of bounds
	MATRIX_ERROR_MEMORY_ALLOCATION, // memory allocation failed
	MATRIX_ERROR_DIMENSION_MISMATCH, // matrix dimensions mismatched
	MATRIX_ERROR_UNKNOWN, // error was not recognized
} MatrixStatus;

/**
* out of bounds
* malloc
* dimension mismatch
*/

// !!TODO look into ways to improve information hiding
typedef struct {
	unsigned int rows;
	unsigned int cols;
	float* elements; // 1d array containing matrix data
} Matrix;

MatrixStatus createMatrix(int rows, int cols, Matrix *out); // throw
void freeMatrix(Matrix mat);
Matrix multiplyMatrix(Matrix a, Matrix b); // throw
Matrix multiplyMatrixElementWise(Matrix a, Matrix b); // throw
void scaleMatrixInPlace(Matrix mat, float scalar);
Matrix scaleMatrix(Matrix mat, float scalar);
MatrixStatus addMatrixInPlace(Matrix a, Matrix b); // throw
Matrix addMatrix(Matrix a, Matrix b); // throw
MatrixStatus subtractMatrixInPlace(Matrix a, Matrix b); // throw
Matrix subtractMatrix(Matrix a, Matrix b); // throw
void fillMatrix(Matrix mat, float value);
Matrix getSubMatrix(Matrix mat, int startRow, int startCol, int numRows, int numCols); // throw
Matrix transposeMatrix(Matrix mat);
void printMatrix(Matrix mat);
Matrix copyMatrix(Matrix src);
float sumMatrix(Matrix src);

#endif // MATRIX_H