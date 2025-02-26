#include "Matrix.h"
#include <stdlib.h>

Matrix createMatrix(int rows, int cols) {
	Matrix mat;
	
	mat.rows = rows;
	mat.cols = cols;
	mat.elements = (double*)malloc(rows * cols * sizeof(double));
	if (mat.elements == NULL)throw "createMatrix: malloc failed";

	return mat;
}

void freeMatrix(Matrix mat) {
	mat.rows = 0;
	mat.cols = 0;
	free(mat.elements);

}

Matrix multiplyMatrix(Matrix a, Matrix b) {
	if (a.cols != b.rows) {
		throw "multiply: Matrix dimensions do not match for multiplication.";
	}
	Matrix result = createMatrix(a.rows, b.cols);

	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < b.cols; j++) {
			for (int k = 0; k < a.cols; k++) {
				result.elements[i*result.cols+j] += a.elements[i * a.cols + k] * b.elements[k * b.cols + j];
			}
		}
	}
	return result;
}

Matrix multiplyMatrixElementWise(Matrix a, Matrix b) {
	// Check if matrices have the same dimensions
	if (a.rows != b.rows || a.cols != b.cols) {
		throw "multiplyElementWise: Matrices must have the same dimensions for element-wise multiplication.";
	}
	Matrix result = createMatrix(a.rows, a.cols);
	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			result.elements[i * result.cols + j] = a.elements[i * a.cols + j] * b.elements[i * b.cols + j];
		}
	}
	return result;
}

void scaleMatrixInPlace(Matrix mat, double scalar) {
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			mat.elements[i * mat.cols + j] *= scalar;
		}
	}
}

Matrix scaleMatrix(Matrix mat, double scalar) {
	Matrix result = createMatrix(mat.rows, mat.cols);
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			result.elements[i * result.cols + j] = mat.elements[i * mat.cols + j] * scalar;
		}
	}
	return result;
}

void addMatrixInPlace(Matrix a, Matrix b) {
	if (a.rows != b.rows || a.cols != b.cols) {
		throw "addInPlace: Matrix dimensions must match for addition.";
	}

	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			a.elements[i * a.cols + j] += b.elements[i * b.cols + j];
		}
	}
}

Matrix addMatrix(Matrix a, Matrix b){
	if (a.rows != b.rows || a.cols != b.cols) throw "add: Matrix dimensions must match for addition.";

	Matrix result = createMatrix(a.rows, a.cols);

	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			result.elements[i * result.cols + j] = b.elements[i * b.cols + j];
		}
	}
	return result;
}

void subtractMatrixInPlace(Matrix a, Matrix b) {
	if (a.rows != b.rows || a.cols != b.cols) {
		throw "subtractInPlace: Matrix dimensions must match for subtraction.";
	}

	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			a.elements[i * a.cols + j] -= b.elements[i * b.cols + j];
		}
	}
}

Matrix subtractMatrix(Matrix a, Matrix b) {
	if (a.rows != b.rows || a.cols != b.cols) throw "subtract: Matrix dimensions must match for subtraction.";

	Matrix result = createMatrix(a.rows, a.cols);

	for (int i = 0; i < a.rows; i++) {
		for (int j = 0; j < a.cols; j++) {
			result.elements[i * result.cols + j] = b.elements[i * b.cols + j];
		}
	}
	return result;
}

void fillMatrix(Matrix mat, double value) {
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			mat.elements[i * mat.cols + j] = value;
		}
	}
}

Matrix getSubMatrix(Matrix mat, int startRow, int startCol, int numRows, int numCols) {
	if (startRow < 0 || startRow + numRows > mat.rows || startCol < 0 || startCol + numCols > mat.cols) {
		throw "getSubMatrix: Submatrix dimensions are out of bounds.";
	}

	Matrix subMatrix = createMatrix(numRows, numCols);
	for (int i = 0; i < numRows; i++) {
		for (int j = 0; j < numCols; j++) {
			subMatrix.elements[i * subMatrix.cols + j] = mat.elements[startRow + i * mat.cols + j + startCol];
		}
	}
	return subMatrix;
}

Matrix transposeMatrix(Matrix mat) {
	Matrix transposed = createMatrix(mat.cols, mat.rows);

	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			transposed.elements[j * transposed.cols + i] = mat.elements[i * mat.cols + j];
		}
	}
	return transposed;
}
