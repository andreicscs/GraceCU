#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
	int rows;
	int cols;
	double* elements; // 1d array containing matrix data
} Matrix;

Matrix createMatrix(int rows, int cols);
void freeMatrix(Matrix mat);
Matrix multiplyMatrix(Matrix a, Matrix b);
Matrix multiplyMatrixElementWise(Matrix a, Matrix b);
void scaleMatrixInPlace(Matrix mat, double scalar);
Matrix scaleMatrix(Matrix mat, double scalar);
void addMatrixInPlace(Matrix a, Matrix b);
Matrix addMatrix(Matrix a, Matrix b);
void subtractMatrixInPlace(Matrix a, Matrix b);
Matrix subtractMatrix(Matrix a, Matrix b);
void fillMatrix(Matrix mat, double value);
Matrix getSubMatrix(Matrix mat, int startRow, int startCol, int numRows, int numCols);
Matrix transposeMatrix(Matrix mat);


#endif // MATRIX_H