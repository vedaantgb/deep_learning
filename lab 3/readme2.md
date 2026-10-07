# MNIST CNN Classifier Execution Summary

This project involves training a Convolutional Neural Network (CNN) on the MNIST handwritten digits dataset using TensorFlow and Keras.

---

## 📊 Summary of Execution

- **Dataset:** MNIST handwriting dataset downloaded via TensorFlow datasets.
- **Model Training:** Trained a CNN model across **5 epochs** on 60,000 training images ($1875 \text{ batches} \times 32 \text{ batch size}$).
- **Final Metrics:**
  - **Final Test Accuracy:** **99.02%**
  - **Final Test Loss:** **0.0313**

---

## 📈 Training Progress

| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
|-------|---------------|-------------------|-----------------|---------------------|
| 1     | 0.1472        | 95.56%            | 0.0531          | 98.14%              |
| 2     | 0.0465        | 98.54%            | 0.0427          | 98.58%              |
| 3     | 0.0326        | 98.98%            | 0.0366          | 98.71%              |
| 4     | 0.0248        | 99.19%            | 0.0315          | 98.99%              |
| 5     | 0.0195        | 99.38%            | 0.0313          | 99.02%              |

---

## ⚠️ Key Observations & Recommended Fixes

### Deprecated Keras Syntax Warning
* **Warning Log:** 
  > `UserWarning: Do not pass an input_shape/input_dim argument to a layer. When using Sequential models, prefer using an Input(shape) object as the first layer in the model instead.`
* **Cause:** In modern Keras (Keras 3), passing `input_shape` inside a feature layer (such as `Conv2D`) is deprecated.
* **Solution:** Explicitly define an `Input` layer as the first element in your model pipeline:

```python
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

---

## 🖼 Plots Rendered
* **Loss & Accuracy Curves:** Generated a 2-axes figure plotting both training vs. validation loss and training vs. validation accuracy over the 5 epochs.