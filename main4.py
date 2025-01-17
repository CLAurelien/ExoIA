import math
import random
import shutil
from mnist import MNIST
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    sig = 1 / (1 + math.exp(-x))
    return sig

poidsCaches = [] # Tableau de 100 tableaux (avec 784 entrées)
poidsFinaux = [] # Tableau de 10 tableaux (avec 100 entrées)
for neurone in range(0, 100):
    poidsCaches.append([])
    for entree in range(0, 784):
        poidsCaches[neurone].append(random.randint(0, 100) / 100 / 784)

for neurone in range(0, 10):
    poidsFinaux.append([])
    for entree in range(0, 100):
        poidsFinaux[neurone].append(random.randint(0, 100) / 100 / 100)

mnist_data = MNIST('./mnist_data')

x_train, y_train = mnist_data.load_training()  # Données d'entraînement
x_test, y_test = mnist_data.load_testing()     # Données de test

# Afficher quelques informations
# print(f"Nombre d'images d'entraînement : {len(x_train)}")
# print(f"Nombre d'images de test : {len(x_test)}")
# print(f"Exemple d'image : {x_train[0]}")
# print(f"Label correspondant : {y_train[0]}")

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

# Normaliser les données
x_train = np.array(x_train) / 255.0
x_test = np.array(x_test) / 255.0

# Reshape des images en format 28x28 si nécessaire
x_train = x_train.reshape(-1, 28, 28)
x_test = x_test.reshape(-1, 28, 28)

def label_to_one_hot(label):
    # Créer un tableau de 10 zéros
    one_hot = [0] * 10
    # Mettre à 1 l'index correspondant au label
    one_hot[label] = 1
    return one_hot

def apprentissage():
    i = 0
    listErreurTotale = []
    erreurTotal = 1
    erreurOne = 1
    erreurZero = 0
    epsilon = 0.01
    while(erreurTotal>0.01):
        potI = []
        xI = []
        deltaCaches = []
        deltaFinaux = []
        sigmaFinaux = []

        nbimage = random.randint(0, 59000)
        imageYd, labelYd = training_data[nbimage]
        entrees = [] # 28x28

        yd = label_to_one_hot(labelYd)

        for neurone in range(0,100):
            for entree in range(0,784):
                entrees[entree] = imageYd[entree]/255
                potI[neurone] += entrees[entree]*poidsCaches[neurone][entree]

            print("Poti : " + str(potI[neurone]))
            xI[neurone] = sigmoid(potI[neurone])


        # Partie des neurones finaux (puisque gestion d'erreurs)
        potI = []
        for neurone in range(0,10):
            for entree in range(0,100):
                entrees[entree] = imageYd[entree]/255
                potI[neurone] += xI[neurone]*poidsFinaux[neurone][entree]

            print("Poti : " + str(potI[neurone]))
            xI[neurone] = sigmoid(potI[neurone])


            print("Result : " + str(xI))
            deltaFinaux[neurone] = (sigmoid(potI[neurone]) * (1 - sigmoid(potI[neurone]))) * yd[neurone] - xI[neurone]

            for entree in range(0, 100):
                sigmaFinaux[neurone] += deltaFinaux[neurone] * poidsFinaux[neurone][entree]

        for neuroneCaches in range(0, 100):
            for neuroneFinaux in range(0, 10):
                deltaCaches[neuroneCaches] = (potI[neuroneCaches] * (1 - potI[neuroneCaches])) * sigmaFinaux[neuroneFinaux]

        for neurone in range(0,10):
            poidsFinaux[neurone] += epsilon * deltaFinaux[neurone] * xI[neurone]

        for neurone in range(0,100):
            poidsCaches[neurone] += epsilon * deltaCaches[neurone] * entrees[neurone]


        # Calculer les erreurs

        if i%2 == 0:
            erreurZero = delta
        else:
            erreurOne = delta

        erreurTotal = abs(erreurOne) + abs(erreurZero)
        listErreurTotale.append(erreurTotal)
        print("Erreur Totale : " + str(erreurTotal))
        i += 1
        print("")
    print("Fin apprentissage")

