#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
	int rows;
	int cols;
	float* elements; // 1d array containing matrix data
} Matrix;

Matrix createMatrix(int rows, int cols);
void freeMatrix(Matrix mat);
Matrix multiplyMatrix(Matrix a, Matrix b);
Matrix multiplyMatrixElementWise(Matrix a, Matrix b);
void scaleMatrixInPlace(Matrix mat, float scalar);
Matrix scaleMatrix(Matrix mat, float scalar);
void addMatrixInPlace(Matrix a, Matrix b);
Matrix addMatrix(Matrix a, Matrix b);
void subtractMatrixInPlace(Matrix a, Matrix b);
Matrix subtractMatrix(Matrix a, Matrix b);
void fillMatrix(Matrix mat, float value);
Matrix getSubMatrix(Matrix mat, int startRow, int startCol, int numRows, int numCols);
Matrix transposeMatrix(Matrix mat);
void printMatrix(Matrix mat);
Matrix copyMatrix(Matrix src);
float sumMatrix(Matrix src);

#endif // MATRIX_H