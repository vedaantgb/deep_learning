# Lab 5 — Handwritten Digit Recognition with an RNN (LSTM)
**Course:** Deep Learning Laboratory (`23DS5PCDLG`)

---

## 📌 Overview
This repository contains the completed Jupyter Notebook for **Lab 5**, which demonstrates how to perform image classification using a **Recurrent Neural Network (RNN / LSTM)** rather than a standard Multi-Layer Perceptron (MLP) or Convolutional Neural Network (CNN).

### **The Core Idea**
Instead of flattening a `28x28` MNIST digit image into a single `784`-dimensional vector, we treat the image as a **sequence of 28 rows** across **28 time steps**. Each row contains **28 pixel values (features)**. 

An **LSTM (Long Short-Term Memory)** network reads the image row-by-row, maintains spatial temporal memory, and predicts the digit class (`0–9`) after reading the final row.

---

## 🚀 Model Architecture & Performance
* **Input Shape:** `(28, 28)` — 28 time steps, 28 features per step.
* **LSTM Layer:** 128 units.
* **Output Layer:** Dense (10 units, `softmax` activation).
* **Optimizer:** Adam
* **Loss Function:** Categorical Cross-Entropy
* **Batch Size:** 64
* **Epochs:** 10
* **Target Test Accuracy:** **~98.72%**

---

## 🛠 Notebook Structure & Tasks

### **Task 1: Data Preparation & Preprocessing**
* Loaded the MNIST dataset.
* Normalized pixel values from `[0, 255]` to `[0.0, 1.0]`.
* Explicitly reshaped dataset tensors to sequence structure `(samples, 28, 28)`.
* One-hot encoded digit labels into 10-dimensional target vectors.

### **Task 2: Model Architecture & Parameter Math**
* Constructed a Keras `Sequential` model with an `LSTM` layer (128 units) and a `Dense` softmax output layer.
* **Parameter Calculation for LSTM Layer:**
  $$\text{Params} = 4 \times [(\text{features} + \text{units}) \times \text{units} + \text{units}]$$
  $$\text{Params} = 4 \times [(28 + 128) \times 128 + 128] = \mathbf{80,384 \text{ parameters}}$$

### **Task 3: Model Compilation & Training**
* Compiled using `optimizer='adam'` and `loss='categorical_crossentropy'`.
* Trained for 10 epochs, achieving **98.72% test accuracy**.

### **Task 4: Training Visualization**
* Plotted training vs. validation accuracy curves across all 10 epochs using `matplotlib`.

### **Task 5: Lab Conceptual Questions & Review**
1. **Feeding `(28, 28)` vs. `784` Vector:** Feeding `(28, 28)` treats the image as 28 sequential steps, preserving row-by-row spatial relationships rather than destroying them via flattening.
2. **Hidden State ($h_t$) vs. Cell State ($C_t$):** $C_t$ acts as the long-term memory conveyor belt, while $h_t$ serves as the short-term working state and output at the current step.
3. **Softmax vs. Sigmoid:** Softmax outputs a normalized probability distribution where all classes sum to 1, perfect for mutually exclusive multi-class digit classification.
4. **Changing `LSTM(128)` to `LSTM(32)`:** Parameter count drops from **80,384** down to **7,808**, significantly speeding up compute at the expense of model memory capacity.
5. **Activation (`tanh` vs. `relu`):** `tanh` is recommended for LSTMs because its bounded range ($[-1, 1]$) prevents recursive hidden states and gradients from exploding.

---

## 💻 Requirements & Environment
* Python `3.10+`
* TensorFlow / Keras `2.x`
* NumPy
* Matplotlib

To run the notebook locally:
```bash
pip install tensorflow numpy matplotlib jupyter
jupyter notebook
```