import random
import shutil
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

mnist_data = MNIST('./mnist_data')

x_train, y_train = mnist_data.load_training()  # Données d'entraînement
x_test, y_test = mnist_data.load_testing()     # Données de test

# Afficher quelques informations
print(f"Nombre d'images d'entraînement : {len(x_train)}")
print(f"Nombre d'images de test : {len(x_test)}")
print(f"Exemple d'image : {x_train[0]}")
print(f"Label correspondant : {y_train[0]}")



# Combiner images et labels dans des tuples
training_data = list(zip(x_train, y_train))
testing_data = list(zip(x_test, y_test))

# Exemple d'accès à un élément
imageYd, labelYd = training_data[42]
print(f"Label : {labelYd}")
print(f"Image (premiers pixels) : {imageYd[:10]}")



# Convertir une image en 2D
image_2d = np.array(imageYd).reshape(28, 28)

# Afficher l'image
plt.imshow(image_2d, cmap='gray')
plt.title(f"Label : {labelYd}")
plt.axis('off')
plt.show()
