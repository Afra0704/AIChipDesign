import os
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Dense, Dropout,
    BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger, ModelCheckpoint

np.random.seed(10)
random.seed(10)
tf.random.set_seed(10)

# =========================
# STEP 1. Prepare dataset
# =========================
(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()

print("Original train shape:", x_train_full.shape)   # (50000, 32, 32, 3)
print("Original test shape:", x_test.shape)          # (10000, 32, 32, 3)
print("Original y_train shape:", y_train_full.shape) # (50000, 1)
print("Original y_test shape:", y_test.shape)        # (10000, 1)

# Normalize pixel values to [0, 1]
x_train_full = x_train_full.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Convert labels from shape (n,1) to (n,)
y_train_full = y_train_full.reshape(-1)
y_test = y_test.reshape(-1)

# Shuffle before splitting validation set
indices = np.arange(len(x_train_full))
np.random.shuffle(indices)
x_train_full = x_train_full[indices]
y_train_full = y_train_full[indices]

# Split training data into train and validation
(x_train, y_train), (x_val, y_val) = (
    (x_train_full[:40000], y_train_full[:40000]),
    (x_train_full[40000:], y_train_full[40000:]),
)

print("Train shape:", x_train.shape)         # (40000, 32, 32, 3)
print("Validation shape:", x_val.shape)      # (10000, 32, 32, 3)
print("Test shape:", x_test.shape)           # (10000, 32, 32, 3)

# =========================
# STEP 2. Define CNN model
# =========================
data_augmentation = Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomRotation(0.05),
], name="data_augmentation")

model = Sequential([
    Input(shape=(32, 32, 3)),
    data_augmentation,

    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    layers.ReLU(),
    Conv2D(64, (3, 3), padding='same'),
    BatchNormalization(),
    layers.ReLU(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    layers.ReLU(),
    Conv2D(128, (3, 3), padding='same'),
    BatchNormalization(),
    layers.ReLU(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.30),

    Conv2D(256, (3, 3), padding='same'),
    BatchNormalization(),
    layers.ReLU(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.40),

    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.50),
    Dense(10, activation='softmax')
])

model.summary()

# =========================
# STEP 3. Compile model
# =========================
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# STEP 4. Callbacks
# =========================

os.makedirs("outputs", exist_ok=True)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=8,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=1e-5,
    verbose=1
)

checkpoint = ModelCheckpoint(
    "outputs/best_model.keras",
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

csv_logger = CSVLogger('outputs/training_history.csv', append=False)

# Save model summary
with open("outputs/model_summary.txt", "w", encoding="utf-8") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# =========================
# STEP 5. Fit model
# =========================
history = model.fit(
    x_train, y_train,
    validation_data=(x_val, y_val),
    epochs=40,
    batch_size=64,
    callbacks=[early_stopping, reduce_lr, csv_logger, checkpoint],
    verbose=1
)

# =========================
# STEP 6. Evaluate model
# =========================
train_loss, train_acc = model.evaluate(x_train, y_train, verbose=0)
val_loss, val_acc = model.evaluate(x_val, y_val, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)

print("Train Accuracy:", train_acc)
print("Validation Accuracy:", val_acc)
print("Test Accuracy:", test_acc)

with open("outputs/final_result.txt", "w", encoding="utf-8") as f:
    f.write(f"Train Accuracy: {train_acc:.6f}\n")
    f.write(f"Validation Accuracy: {val_acc:.6f}\n")
    f.write(f"Train Loss: {train_loss:.6f}\n")
    f.write(f"Validation Loss: {val_loss:.6f}\n")
    f.write(f"Test Loss: {test_loss:.6f}\n")
    f.write(f"Test Accuracy: {test_acc:.6f}\n")

model.save("outputs/cifar10_cnn_improved.keras")

# =========================
# Plot Accuracy and Loss
# =========================
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("outputs/accuracy_loss_curve.png", dpi=300)
plt.show()