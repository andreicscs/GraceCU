
// check memory leaks
#define _CRTDBG_MAP_ALLOC
#include <cstdlib>
#include <crtdbg.h>

#ifdef _DEBUG
#define DEBUG_NEW new(_NORMAL_BLOCK, __FILE__, __LINE__)
#define new DEBUG_NEW
#define malloc(s) _malloc_dbg(s, _NORMAL_BLOCK, __FILE__, __LINE__)
#endif

#include "Matrix.h"
#include <stdlib.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


using namespace std;


__global__ void matMulKernel(float* res, float* a, float* b, int resRows, int resCols, int aCols, int bCols);
__global__ void matAdd();
bool isGpuAvailable();



__global__ void matMulKernel(float* res, float* a, float* b, int resRows, int resCols, int aCols, int bCols) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	// if the row and col is in bounds.
	if (row < resRows && col < resCols) {
		float sum = 0.0;
		for (int k = 0; k < aCols; k++) {
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
	Matrix dest = createMatrix(src.rows, src.cols);
	for (int i = 0; i < src.rows * src.cols; ++i) {
		dest.elements[i] = src.elements[i];
	}
	return dest;
}

Matrix createMatrix(int rows, int cols) {
	Matrix mat;
	
	mat.rows = rows;
	mat.cols = cols;
	mat.elements = (float*)malloc(rows * cols * sizeof(float));
	if (mat.elements == NULL)throw "createMatrix: malloc failed";

	return mat;
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
		throw "multiply: Matrix dimensions do not match for multiplication.";
	}

	Matrix result;


	if (true) { // !! TO DO cuda computations are slow because of way too frequent memory operations
		result = createMatrix(a.rows, b.cols);
		fillMatrix(result, 0);

		for (int i = 0; i < a.rows; i++) {
			for (int j = 0; j < b.cols; j++) {
				for (int k = 0; k < a.cols; k++) {
					result.elements[i * result.cols + j] += a.elements[i * a.cols + k] * b.elements[k * b.cols + j];
				}
			}
		}
		return result;
	}
	result = createMatrix(a.rows, b.cols);
	//result = {a.rows, b.cols, nullptr};
	
	float* dA;
	float* dB;
	float* dRes;

	// allocate memory on device
	cudaMalloc(&dA, (size_t)(a.rows * a.cols * sizeof(float)));
	cudaMalloc(&dB, (size_t)(b.rows * b.cols * sizeof(float)));
	cudaMalloc(&dRes, (size_t)(result.rows * result.cols * sizeof(float)));

	// copy memory to device
	cudaMemcpy(dA, a.elements, a.rows*a.cols * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(dB, b.elements, b.rows * b.cols * sizeof(float), cudaMemcpyHostToDevice);

	// launch kernel
	dim3 blockDim(16,16);
	dim3 gridDim((b.cols + blockDim.x - 1) / blockDim.x, (a.rows + blockDim.y - 1) / blockDim.y); // blockDim - 1 to round up division in case cols and rows are not a multiple of 16
	matMulKernel <<<gridDim, blockDim>>> (dRes, dA, dB, result.rows, result.cols, a.cols, b.cols);


	// copy result back to host
	cudaMemcpy(result.elements, dRes, result.rows * result.cols * sizeof(float), cudaMemcpyDeviceToHost);

	// free memory
	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dRes);

	return result;
}

Matrix multiplyMatrixElementWise(Matrix a, Matrix b) {
	// check if matrices have the same dimensions
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

void scaleMatrixInPlace(Matrix mat, float scalar) {
	for (int i = 0; i < mat.rows; i++) {
		for (int j = 0; j < mat.cols; j++) {
			mat.elements[i * mat.cols + j] *= scalar;
		}
	}
}

Matrix scaleMatrix(Matrix mat, float scalar) {
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
			result.elements[i * result.cols + j] = a.elements[i * a.cols + j] - b.elements[i * b.cols + j];
		}
	}
	return result;
}

void fillMatrix(Matrix mat, float value) {
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
			subMatrix.elements[i * subMatrix.cols + j] = mat.elements[(startRow + i) * mat.cols + (j + startCol)];
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

void printMatrix(Matrix mat) {
	for (int i = 0; i < mat.rows;++i) {
		for (int j = 0; j < mat.cols; ++j) {
			cout << mat.elements[i * mat.cols + j]<<" ";
		}
		cout << endl;
	}
}

float sumMatrix(Matrix src) {
	float result = 0.0;
	for (int i = 0; i < src.rows; ++i) {
		for (int j = 0; j < src.cols; ++j) {
			result += src.elements[i * src.cols + j];
		}
	}
	return result;
}