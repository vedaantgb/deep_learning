# Covertype Forest Cover Type Prediction

This repository contains a Deep Feedforward Neural Network (DFNN) model built using **TensorFlow / Keras** to predict forest cover types from cartographic variables using the **Covertype Dataset** from `scikit-learn`.

## 📌 Project Overview

The Forest Covertype dataset contains 581,012 samples with 54 features representing environmental and cartographic attributes (such as elevation, aspect, slope, soil type, and wilderness areas). The goal is to classify each $30 \times 30$ meter cell into one of 7 forest cover types.

### Key Challenges Handled
1. **Severe Class Imbalance**: Imbalance ratio between the most frequent and least frequent class is **103.13**[cite: 1]. Handled using `compute_class_weight` during model training[cite: 1].
2. **Feature Heterogeneity**: First 10 features are continuous, while the remaining 44 features are binary/one-hot encoded[cite: 1]. Continuous features are standardized using `StandardScaler`[cite: 1].
3. **Multi-Class Target Alignment**: Targets are shifted from 1–7 to 0–6 to conform to standard 0-indexed sparse categorical cross-entropy loss[cite: 1].

---

## 📊 Dataset Summary

* **Total Samples**: 581,012[cite: 1]
* **Number of Features**: 54 (10 continuous, 44 binary)[cite: 1]
* **Target Classes**: 7 classes (Cover Type 1 to 7)[cite: 1]
* **Split Ratio**:
  * **Train Set**: 80% (464,809 samples)[cite: 1]
  * **Validation Set**: 10% (58,101 samples)[cite: 1]
  * **Test Set**: 10% (58,102 samples)[cite: 1]

---

## 🏗️ Model Architecture

The Deep Feedforward Neural Network (DFNN) uses a sequential architecture `[54 ➔ 64 ➔ 32 ➔ 16 ➔ 7]`:

| Layer | Type | Activation | Output Shape | Param # |
| :--- | :--- | :--- | :--- | :--- |
| `hidden_layer_1` | Dense | ReLU | `(None, 64)` | 3,520 |
| `hidden_layer_2` | Dense | ReLU | `(None, 32)` | 2,080 |
| `hidden_layer_3` | Dense | ReLU | `(None, 16)` | 528 |
| `output_layer` | Dense | Softmax | `(None, 7)` | 119 |

* **Total Parameters**: 6,247 (24.40 KB)
* **Optimizer**: Adam
* **Loss Function**: `sparse_categorical_crossentropy`
* **Metrics**: `accuracy`

---

## ⚙️ Requirements

To run this notebook, ensure you have the following dependencies installed:

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib
```