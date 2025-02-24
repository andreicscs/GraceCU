#ifndef MATRIX_H
#define MATRIX_H

typedef struct {
	int rows;
	int cols;
	double* elements; // 1d array containing matrix data
} Matrix;

Matrix createMatrix(int rows, int cols);
void freeMatrix(Matrix mat);
Matrix multiply(Matrix a, Matrix b);
Matrix multiplyElementWise(Matrix a, Matrix b);
void scaleInPlace(Matrix mat, double scalar);
Matrix scale(Matrix mat, double scalar);
void addInPlace(Matrix a, Matrix b);
Matrix add(Matrix a, Matrix b);
void subtractInPlace(Matrix a, Matrix b);
Matrix subtract(Matrix a, Matrix b);
void fillMatrix(Matrix mat, double value);
Matrix getSubMatrix(Matrix mat, int startRow, int startCol, int numRows, int numCols);
Matrix transpose(Matrix mat);


#endif // MATRIX_H