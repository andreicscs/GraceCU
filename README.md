[![License](https://img.shields.io/badge/license-GPLv3-blue.svg)](LICENSE)
[![C](https://img.shields.io/badge/language-C-brightgreen.svg)]()
[![Status](https://img.shields.io/badge/status-WIP-yellow.svg)]()

## 📑 Table of Contents
- [Background & Evolution](#-background--evolution)
- [Project Structure](#-project-structure)
- [Matrix Library](#-matrix-library-matrixh)
- [Neural Network Library](#-neural-network-library-neuralnetworkh)
- [Documentation](#-documentation)
- [Future Plans](#-future-plans)
- [Feedback](#-feedback)

# GraceCU  
## 📘 Background & Evolution  
This project started as part of the [GRACE](https://github.com/Fairswing/Grace) learning initiative, which was designed to help students understand the fundamentals of artificial intelligence by implementing neural networks from scratch using backpropagation and gradient descent.  

### 🎓 Academic Purpose  
As the scope grew, the neural network component evolved into a standalone **academic library**. The focus is now on **clarity, modularity, and educational value** rather than raw performance. 
***The library is intended as a learning tool for students who want to understand how neural networks work internally, step by step.*** 
The library includes detailed comments that explain the code’s behavior, linking each implementation to the underlying mathematical formulas and clarifying both what they do and why it works that way.

---

## 🧩 Project Structure  
The project currently consists of two main libraries:  

- **`Matrix.h`** → A custom matrix library implementing core operations.
- **`NeuralNetwork.h`** → A neural network library built on top of `Matrix.h`.  
### 📊 MNIST Demo  
A demo application is provided to train and test a simple neural network on the **MNIST digit recognition dataset**, demonstrating the use of `Matrix.h` and `NeuralNetwork.h` in practice.  

---

## 🧮 Matrix Library (`Matrix.h`)  

### Features  
- Matrix creation and deletion  
- Basic arithmetic operations (addition, subtraction, multiplication)  
- Element-wise operations  
- Scaling and transposition  
- Matrix storage and loading from files  

### TODO  
- Write unit tests  
- Improve documentation  
- Optimize matrix multiplication  
- (Optional) Reintroduce CUDA kernels for GPU acceleration in the future  

---

## 🧠 Neural Network Library (`NeuralNetwork.h`)  

### Features  
- Dynamic network architecture  
- Activation functions: Sigmoid, ReLU, Softmax  
- Loss functions: CCE, MSE, BCE  
- Training with batch processing  
- Model persistence (save/load network states)  
- Accuracy and loss computation
- Nural network prediction (to use the trained neural network)

### TODO  
- Expand test coverage (known-answer tests or test helpers)  
- Improve documentation (full API docs, install/run instructions)  
- Add regularization and optimizers (Adam, RMSProp, etc.)  
- Add support for dropout, batch normalization, and convolutional layers  
- Provide more didactic examples and exercises  

---

## 📘 Documentation
For a full API reference and usage details, check out the [📖 Documentation](./docs/API.md).

---

## 🚀 Future Plans  
- Expand the library with additional layers and functions  
- Provide exercises/tutorials for students  
- Improve ease of use with data loading and data manipulation functions
- (Optional) Revisit CUDA or multithreading if performance becomes a focus  

## 🤝 Feedback  
This project is still under development. Contributions, suggestions, or feedback from students and educators are very welcome! 

---


Copyright (C) 2025 Termine Andrea

This program is free software: you can redistribute it and/or modify it
under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program. If not, see <https://www.gnu.org/licenses/>.
