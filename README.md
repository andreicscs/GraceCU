# Work in Progress

## 📘 Background & Evolution
This project originally began as part of the [GRACE](https://github.com/Fairswing/Grace) learning initiative, which was created for educational purposes to explore the fundamentals of artificial intelligence and how to implement a basic neural network from scratch using backpropagation and gradient descent.
However, as the scope grew, the neural network component evolved into a standalone, general-purpose neural network library, independent from the original GRACE application. The codebase now focuses on building a performant, modular deep learning library with extensible features.


---

This project is currently a work in progress. It consists of two main libraries:

- `Matrix.h`: A custom matrix library that implements basic matrix operations.
- `NeuralNetwork.h`: A neural network library built from scratch that utilizes `Matrix.h` for computations.

Additionally, a `main` function and an MNIST loader have been implemented as a demo to showcase the functionality of the neural network on the MNIST digit recognition dataset.

---

## Matrix Library (`Matrix.h`)

### Features:
- Matrix creation and deletion
- Basic arithmetic operations (addition, subtraction, multiplication)
- Element-wise operations
- Scaling and transposition
- Matrix storage and loading from files

### TODO:
- Write tests.
- Write documentation.
- Implement CUDA kernels for GPU-parallelized computations.
- Improve information hiding.
- Implement more efficient matrix multiplication.

---

## Neural Network Library (`NeuralNetwork.h`)

### Features:
- Customizable network architecture
- Various activation functions (Sigmoid, ReLU, Softmax)
- Different loss functions (CCE, MSE, BCE)
- Training with batch processing
- Model persistence (saving and loading network states)
- Accuracy and loss computation

### TODO:
- Write tests. Since the API is quite restrictive, the options are: Known Answer Tests to test public APIs, or implementing a "test helper" header that makes all functions and structs that need to be tested public for easier and more in-depth testing.
- Improve documentation. Write a README with complete API documentation and instructions on how to install and run the project.
- Implement data loading and processing functions (consider taking `nn_config` as a parameter).
- Implement regularization.
- Implement optimizers. (e.g., Adam, RMSprop)
- Implement CUDA kernels for GPU-parallelized computations.
- Implement support for dropout and batch normalization.
- Expand available activation and loss functions.
- Add support for convolutional layers.
- Add
---

## MNIST Demo

A demo application is provided to showcase the functionality of `Matrix.h` and `NeuralNetwork.h` by training and testing a neural network on the MNIST digit recognition dataset.

---

## Future Plans
- Expand the neural network library to support more complex models.
- Optimize performance for large-scale datasets.
- Provide better documentation and examples for users.

This project is still under development, feedback is very welcome!
