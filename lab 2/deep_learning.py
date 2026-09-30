import numpy as np
import tensorflow as tf
from sklearn.datasets import make_moons
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD

# ---- Data Preparation ----
X, y = make_moons(n_samples=400, noise=0.20, random_state=1)
y = y.reshape(-1, 1).astype(float)
X = (X - X.mean(0)) / X.std(0)  # standardize

# Helper to build a fresh, un-trained model architecture
def build_model(learning_rate):
    model = Sequential([
        Input(shape=(2,)),
        Dense(16, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        loss='binary_crossentropy',
        optimizer=SGD(learning_rate=learning_rate),
        metrics=['accuracy']
    )
    return model

# ---- 1. Batch Gradient Descent (Full dataset in 1 batch) ----
tf.random.set_seed(0)
model_bgd = build_model(learning_rate=0.5)
model_bgd.fit(X, y, epochs=200, batch_size=len(X), verbose=0)
loss_bgd, acc_bgd = model_bgd.evaluate(X, y, verbose=0)
print(f"Batch GD Accuracy : {acc_bgd:.4f}")

# ---- 2. Mini-Batch SGD (Batch size = 16) ----
tf.random.set_seed(0)
model_sgd = build_model(learning_rate=0.1)  # Lower learning rate for mini-batches
model_sgd.fit(X, y, epochs=200, batch_size=16, verbose=0)
loss_sgd, acc_sgd = model_sgd.evaluate(X, y, verbose=0)
print(f"SGD Accuracy      : {acc_sgd:.4f}")

