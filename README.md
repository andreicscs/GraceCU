# GraceCU  
## 📘 Background & Evolution  
This project started as part of the [GRACE](https://github.com/Fairswing/Grace) learning initiative, which was designed to help students understand the fundamentals of artificial intelligence by implementing neural networks from scratch using backpropagation and gradient descent.  

As the scope grew, the neural network component evolved into a standalone **academic library**. The focus is now on **clarity, modularity, and educational value** rather than raw performance. The library is intended as a learning tool for students who want to understand how neural networks work internally, step by step.  

---

## 🧩 Project Structure  
The project currently consists of two main libraries:  

- **`Matrix.h`** → A custom matrix library implementing core operations.  
- **`NeuralNetwork.h`** → A neural network library built on top of `Matrix.h`.  

A simple `main` demo and an **MNIST loader** are included to showcase training on the MNIST digit recognition dataset.  

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
- Customizable network architecture  
- Activation functions: Sigmoid, ReLU, Softmax  
- Loss functions: CCE, MSE, BCE  
- Training with batch processing  
- Model persistence (save/load network states)  
- Accuracy and loss computation  

### TODO  
- Expand test coverage (known-answer tests or test helpers)  
- Improve documentation (full API docs, install/run instructions)  
- Add regularization and optimizers (Adam, RMSProp, etc.)  
- Add support for dropout, batch normalization, and convolutional layers  
- Provide more didactic examples and exercises  

---

## 🎓 Academic Purpose  
Unlike production frameworks such as TensorFlow or PyTorch, **GraceCU is not optimized for maximum performance**. Instead, it prioritizes:  
- Clear code structure  
- Readability and learning value  
- Step-by-step understanding of matrix math and backpropagation  
- Serving as a teaching/learning tool in academic contexts  

---

## 📊 MNIST Demo  
A demo application is provided to train and test a simple neural network on the **MNIST digit recognition dataset**, demonstrating the use of `Matrix.h` and `NeuralNetwork.h` in practice.  

---

## 🚀 Future Plans  
- Expand the library with additional layers and functions  
- Provide exercises/tutorials for students  
- Improve ease of use with helper utilities  
- (Optional) Revisit CUDA or multithreading if performance becomes a focus  

---

## 🤝 Feedback  
This project is still under development. Contributions, suggestions, or feedback from students and educators are very welcome!  
