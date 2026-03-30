import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
import random

np.random.seed(10)
random.seed(10)

# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x = image, y = label

print('train data shape =', x_train.shape)   # (60000, 28, 28)
print('test data shape =', x_test.shape)     # (10000, 28, 28)

#print("x_train dtype =", x_train.dtype)
#print("x_train min =", x_train.min(), ", max =", x_train.max())

# Normalize the pixel values to the range [0, 1]
x_train_normalize = x_train.astype("float32") / 255.0
x_test_normalize = x_test.astype("float32") / 255.0

#print("x_train_normalize dtype =", x_train_normalize.dtype)
#print("x_train_normalize min =", x_train_normalize.min(), ", max =", x_train_normalize.max())

# Show some random images from the training set
fig, axs = plt.subplots(1, 6, figsize=(15, 3))
for i in range(6):
    idx = random.randint(0, len(x_train_normalize) - 1)
    img, label = x_train_normalize[idx], y_train[idx]
    axs[i].imshow(img, cmap='gray')
    axs[i].set_title(f'Label: {label}')
    axs[i].axis('off')

plt.tight_layout()
plt.show()