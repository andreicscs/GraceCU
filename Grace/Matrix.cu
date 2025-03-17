// check memory leaks
#define _CRTDBG_MAP_ALLOC
#include <crtdbg.h>

#ifdef _DEBUG
#define malloc(s) _malloc_dbg(s, _NORMAL_BLOCK, __FILE__, __LINE__)
#endif

#include <stdlib.h>
#include <stdio.h>
#include "Matrix.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/**
* TODO list:
*   implement error handling:
*   implement cuda kernels for gpu parallelied computations.
*
*/

__global__ void matMulKernel(float* res, float* a, float* b, int resRows, int resCols, int aCols, int bCols);
__global__ void matAdd();
bool isGpuAvailable();



__global__ void matMulKernel(float* res, float* a, float* b, int resRows, int resCols, int aCols, int bCols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	// if the row and col is in bounds.
	if (row < resRows && col < resCols) {
		float sum = 0.0;
		for (unsigned int k = 0; k < aCols; k++) {
			sum += a[row * aCols + k] * b[k * bCols + col];
		}
		res[row * resCols + col] = sum;
	}
}

__global__ void matAdd() {

}


bool isGpuAvailable() {
	int deviceCount = 0;
	cudaError_t error = cudaGetDeviceCount(&deviceCount);
	if (error != cudaSuccess || deviceCount == 0) {
		return false;
	}
	return true;
}

Matrix copyMatrix(Matrix src) {
	Matrix dest;
	MatrixStatus err = createMatrix(src.rows, src.cols, &dest); //if (err != MATRIX_OK)return err;
	for (unsigned int i = 0; i < src.rows * src.cols; ++i) {
		dest.elements[i] = src.elements[i];
	}
	return dest;
}

MatrixStatus createMatrix(int rows, int cols, Matrix* out) {
	Matrix mat;
	
	mat.rows = rows;
	mat.cols = cols;
	mat.elements = (float*)malloc(rows * cols * sizeof(float));
	if (mat.elements == NULL) return MATRIX_ERROR_MEMORY_ALLOCATION;

	*out = mat;
	return MATRIX_OK;
}

void freeMatrix(Matrix mat) {
	mat.rows = 0;
	mat.cols = 0;
	if (mat.elements != nullptr) {
		free(mat.elements);
	}
}

Matrix multiplyMatrix(Matrix a, Matrix b) {
	if (a.cols != b.rows) {
		//return MATRIX_ERROR_DIMENSION_MISMATCH;
	}
	Matrix result;
	MatrixStatus err = createMatrix(a.rows, b.cols, &result);
	if (err!=0) {
		//return err;
	}
	fillMatrix(result, 0);

	for (unsigned int i = 0; i < a.rows; i++) {
		for (unsigned int j = 0; j < b.cols; j++) {
			for (unsigned int k = 0; k < a.cols; k++) {
				result.elements[i * result.cols + j] += a.elements[i * a.cols + k] * b.elements[k * b.cols + j];
			}
		}
	}
	return result;
}

Matrix multiplyMatrixElementWise(Matrix a, Matrix b) {
	// check if matrices have the same dimensions
	if (a.rows != b.rows || a.cols != b.cols) {
		//return MATRIX_ERROR_DIMENSION_MISMATCH;
	}
	Matrix result;
	MatrixStatus err = createMatrix(a.rows, a.cols, &result);
	if (err != 0) {
		//return err;
	}
	for (unsigned int i = 0; i < a.rows; i++) {
		for (unsigned int j = 0; j < a.cols; j++) {
			result.elements[i * result.cols + j] = a.elements[i * a.cols + j] * b.elements[i * b.cols + j];
		}
	}
	return result;
}

void scaleMatrixInPlace(Matrix mat, float scalar) {
	for (unsigned int i = 0; i < mat.rows; i++) {
		for (unsigned int j = 0; j < mat.cols; j++) {
			mat.elements[i * mat.cols + j] *= scalar;
		}
	}
}

Matrix scaleMatrix(Matrix mat, float scalar) {
	Matrix result;
	MatrixStatus err = createMatrix(mat.rows, mat.cols, &result);
	if (err != 0) {
		//return err;
	}
	for (unsigned int i = 0; i < mat.rows; i++) {
		for (unsigned int j = 0; j < mat.cols; j++) {
			result.elements[i * result.cols + j] = mat.elements[i * mat.cols + j] * scalar;
		}
	}
	return result;
}

MatrixStatus addMatrixInPlace(Matrix a, Matrix b) {
	if (a.rows != b.rows || a.cols != b.cols) {
		return MATRIX_ERROR_DIMENSION_MISMATCH;
	}

	for (unsigned int i = 0; i < a.rows; i++) {
		for (unsigned int j = 0; j < a.cols; j++) {
			a.elements[i * a.cols + j] += b.elements[i * b.cols + j];
		}
	}
}

Matrix addMatrix(Matrix a, Matrix b){
	if (a.rows != b.rows || a.cols != b.cols); //return MATRIX_ERROR_DIMENSION_MISMATCH;

	Matrix result;
	MatrixStatus err = createMatrix(a.rows, a.cols, &result);
	if (err != 0) {
		//return err;
	}
	for (unsigned int i = 0; i < a.rows; i++) {
		for (unsigned int j = 0; j < a.cols; j++) {
			result.elements[i * result.cols + j] = b.elements[i * b.cols + j];
		}
	}
	return result;
}

MatrixStatus subtractMatrixInPlace(Matrix a, Matrix b) {
	if (a.rows != b.rows || a.cols != b.cols) {
		return MATRIX_ERROR_DIMENSION_MISMATCH;
	}

	for (unsigned int i = 0; i < a.rows; i++) {
		for (unsigned int j = 0; j < a.cols; j++) {
			a.elements[i * a.cols + j] -= b.elements[i * b.cols + j];
		}
	}
}

Matrix subtractMatrix(Matrix a, Matrix b) {
	if (a.rows != b.rows || a.cols != b.cols); //return MATRIX_ERROR_DIMENSION_MISMATCH;

	Matrix result;
	MatrixStatus err = createMatrix(a.rows, a.cols, &result);
	if (err != 0) {
		//return err;
	}
	for (unsigned int i = 0; i < a.rows; i++) {
		for (unsigned int j = 0; j < a.cols; j++) {
			result.elements[i * result.cols + j] = a.elements[i * a.cols + j] - b.elements[i * b.cols + j];
		}
	}
	return result;
}

void fillMatrix(Matrix mat, float value) {
	for (unsigned int i = 0; i < mat.rows; i++) {
		for (unsigned int j = 0; j < mat.cols; j++) {
			mat.elements[i * mat.cols + j] = value;
		}
	}
}

Matrix getSubMatrix(Matrix mat, int startRow, int startCol, int numRows, int numCols) {
	if (startRow < 0 || startRow + numRows > mat.rows || startCol < 0 || startCol + numCols > mat.cols) {
		//return MATRIX_ERROR_OUTOFBOUNDS;
	}

	Matrix subMatrix;
	MatrixStatus err = createMatrix(numRows, numCols, &subMatrix);
	if (err != 0) {
		//return err;
	}
	for (unsigned int i = 0; i < numRows; i++) {
		for (unsigned int j = 0; j < numCols; j++) {
			subMatrix.elements[i * subMatrix.cols + j] = mat.elements[(startRow + i) * mat.cols + (j + startCol)];
		}
	}
	return subMatrix;
}

Matrix transposeMatrix(Matrix mat) {
	Matrix transposed;
	MatrixStatus err = createMatrix(mat.cols, mat.rows, &transposed);
	if (err != 0) {
		//return err;
	}

	for (unsigned int i = 0; i < mat.rows; i++) {
		for (unsigned int j = 0; j < mat.cols; j++) {
			transposed.elements[j * transposed.cols + i] = mat.elements[i * mat.cols + j];
		}
	}
	return transposed;
}

void printMatrix(Matrix mat) {
	for (unsigned int i = 0; i < mat.rows;++i) {
		for (unsigned int j = 0; j < mat.cols; ++j) {
			printf("%f ", mat.elements[i * mat.cols + j]);
		}
		printf("\n");
	}
}

float sumMatrix(Matrix src) {
	float result = 0.0;
	for (unsigned int i = 0; i < src.rows; ++i) {
		for (unsigned int j = 0; j < src.cols; ++j) {
			result += src.elements[i * src.cols + j];
		}
	}
	return result;
}