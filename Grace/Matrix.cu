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
*	write tests.
*   write documentation
*   implement cuda kernels for gpu parallelied computations.
*
*/

__global__ void matMulKernel(float* res, float* a, float* b, int resRows, int resCols, int aCols, int bCols);
__global__ void matAdd();
bool isGpuAvailable();



__global__ void matMulKernel(float* res, float* a, float* b, int resRows, int resCols, int aCols, int bCols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
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
	if(src.elements==MATRIX_invalidP)return { 0,0,MATRIX_invalidP };
	Matrix dest = createMatrix(src.rows, src.cols);
	for (unsigned int i = 0; i < src.rows * src.cols; ++i) {
		dest.elements[i] = src.elements[i];
	}
	return dest;
}

Matrix createMatrix(unsigned int rows, unsigned int cols) {
	Matrix mat;

	mat.rows = rows;
	mat.cols = cols;
	mat.elements = (float*)malloc(rows * cols * sizeof(float));
	if (mat.elements == NULL)return { 0,0,MATRIX_invalidP };

	return mat;
}

void freeMatrix(Matrix mat) {
	mat.rows = 0;
	mat.cols = 0;
	if (mat.elements != MATRIX_invalidP) {
		free(mat.elements);
	}
}

Matrix multiplyMatrix(Matrix a, Matrix b) {
	if (a.cols != b.rows) {
		return { 0,0,MATRIX_invalidP };
	}

	Matrix result = createMatrix(a.rows, b.cols);
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
	if (a.rows != b.rows || a.cols != b.cols) {
		return { 0,0,MATRIX_invalidP };
	}
	Matrix result = createMatrix(a.rows, a.cols);
	for (unsigned int i = 0; i < a.rows; i++) {
		for (unsigned int j = 0; j < a.cols; j++) {
			result.elements[i * result.cols + j] = a.elements[i * a.cols + j] * b.elements[i * b.cols + j];
		}
	}
	return result;
}

bool scaleMatrixInPlace(Matrix mat, float scalar) {
	if (mat.elements == NULL)return false;
	for (unsigned int i = 0; i < mat.rows; i++) {
		for (unsigned int j = 0; j < mat.cols; j++) {
			mat.elements[i * mat.cols + j] *= scalar;
		}
	}
	return true;
}

Matrix scaleMatrix(Matrix mat, float scalar) {
	if (mat.elements == NULL)return { 0,0,MATRIX_invalidP };
	Matrix result = createMatrix(mat.rows, mat.cols);
	for (unsigned int i = 0; i < mat.rows; i++) {
		for (unsigned int j = 0; j < mat.cols; j++) {
			result.elements[i * result.cols + j] = mat.elements[i * mat.cols + j] * scalar;
		}
	}
	return result;
}

bool addMatrixInPlace(Matrix a, Matrix b) {
	if (a.rows != b.rows || a.cols != b.cols) {
		return false;
	}

	for (unsigned int i = 0; i < a.rows; i++) {
		for (unsigned int j = 0; j < a.cols; j++) {
			a.elements[i * a.cols + j] += b.elements[i * b.cols + j];
		}
	}
	return true;
}

Matrix addMatrix(Matrix a, Matrix b) {
	if (a.rows != b.rows || a.cols != b.cols) return { 0,0,MATRIX_invalidP };

	Matrix result = createMatrix(a.rows, a.cols);

	for (unsigned int i = 0; i < a.rows; i++) {
		for (unsigned int j = 0; j < a.cols; j++) {
			result.elements[i * result.cols + j] = b.elements[i * b.cols + j];
		}
	}
	return result;
}

bool subtractMatrixInPlace(Matrix a, Matrix b) {
	if (a.rows != b.rows || a.cols != b.cols) {
		return false;
	}

	for (unsigned int i = 0; i < a.rows; i++) {
		for (unsigned int j = 0; j < a.cols; j++) {
			a.elements[i * a.cols + j] -= b.elements[i * b.cols + j];
		}
	}
	return true;
}

Matrix subtractMatrix(Matrix a, Matrix b) {
	if (a.rows != b.rows || a.cols != b.cols) return { 0,0,MATRIX_invalidP };

	Matrix result = createMatrix(a.rows, a.cols);

	for (unsigned int i = 0; i < a.rows; i++) {
		for (unsigned int j = 0; j < a.cols; j++) {
			result.elements[i * result.cols + j] = a.elements[i * a.cols + j] - b.elements[i * b.cols + j];
		}
	}
	return result;
}

bool fillMatrix(Matrix mat, float value) {
	if (mat.elements == NULL)return false;
	for (unsigned int i = 0; i < mat.rows; i++) {
		for (unsigned int j = 0; j < mat.cols; j++) {
			mat.elements[i * mat.cols + j] = value;
		}
	}
	return true;
}

Matrix getSubMatrix(Matrix mat, unsigned int startRow, unsigned int startCol, unsigned int numRows, unsigned int numCols) {
	if (startRow + numRows > mat.rows || startCol + numCols > mat.cols) {
		return { 0,0,MATRIX_invalidP };
	}

	Matrix subMatrix = createMatrix(numRows, numCols);
	for (unsigned int i = 0; i < numRows; i++) {
		for (unsigned int j = 0; j < numCols; j++) {
			subMatrix.elements[i * subMatrix.cols + j] = mat.elements[(startRow + i) * mat.cols + (j + startCol)];
		}
	}
	return subMatrix;
}

Matrix transposeMatrix(Matrix mat) {
	if (mat.elements == NULL)return { 0,0,MATRIX_invalidP };
	Matrix transposed = createMatrix(mat.cols, mat.rows);

	for (unsigned int i = 0; i < mat.rows; i++) {
		for (unsigned int j = 0; j < mat.cols; j++) {
			transposed.elements[j * transposed.cols + i] = mat.elements[i * mat.cols + j];
		}
	}
	return transposed;
}

void printMatrix(Matrix mat) {
	if (mat.elements == NULL)return;
	for (unsigned int i = 0; i < mat.rows; ++i) {
		for (unsigned int j = 0; j < mat.cols; ++j) {
			printf("%f ", mat.elements[i * mat.cols + j]);
		}
		printf("\n");
	}
}

float sumMatrix(Matrix src) {
	if (src.elements == NULL)return MATRIX_invalidP;
	float result = 0.0;
	for (unsigned int i = 0; i < src.rows; ++i) {
		for (unsigned int j = 0; j < src.cols; ++j) {
			result += src.elements[i * src.cols + j];
		}
	}
	return result;
}

bool storeMatrix(Matrix mat, FILE *fpOut) {
	if (fpOut == NULL) {
		return false;
	}

	size_t writtenElements = fwrite(&mat.rows, sizeof(unsigned int), 1, fpOut);
	if (writtenElements != 1) return false;
	
	writtenElements = fwrite(&mat.cols, sizeof(unsigned int), 1, fpOut);
	if (writtenElements != 1) return false;
	
	size_t elementsToWrite = mat.rows * mat.cols;
	writtenElements = fwrite(mat.elements, sizeof(float), elementsToWrite, fpOut);
	if (writtenElements!= elementsToWrite) return false;
	
	return true;
}

bool loadMatrix(FILE *fpIn, Matrix *out) {
	if (fpIn == NULL) {
		return false;
	}
	
	unsigned int rows, cols;
	size_t readElements = fread(&rows, sizeof(unsigned int), 1, fpIn);
	if (readElements != 1) return false;

	readElements = fread(&cols, sizeof(unsigned int), 1, fpIn);
	if (readElements != 1) return false;

	*out = createMatrix(rows, cols);
	if (out->elements == MATRIX_invalidP)return false;

	size_t elementsToRead = out->rows * out->cols;

	readElements = fread(out->elements, sizeof(float), elementsToRead, fpIn);
	if (readElements != elementsToRead) {
		freeMatrix(*out);
		return false;
	}

	return true;
}