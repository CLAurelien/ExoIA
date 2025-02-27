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
# print(f"Label : {labelYd}")
# print(f"Image (premiers pixels) : {imageYd[:10]}")

# Convertir une image en 2D
image_2d = np.array(imageYd).reshape(28, 28)

# Afficher l'image
# plt.imshow(image_2d, cmap='gray')
# plt.title(f"Label : {labelYd}")
# plt.axis('off')
# plt.show()

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
    p = 10000
    listErreurTotale = [0] * p
    epsilon = 0.01
    for i in range(0,p):
        print(i)
        erreurTotal = 1
        potI = [0] * 100
        xH = [0] * 100
        xI = [0] * 10
        deltaCaches = [0] * 100
        deltaFinaux = [0] * 10
        sigmaFinaux = [0] * 10

        nbimage = random.randint(0, 59000)
        imageYd, labelYd = training_data[nbimage]
        entrees = [0] * (28*28) # 28x28

        yd = label_to_one_hot(labelYd)

        for neurone in range(0,100):
            for entree in range(0,784):
                entrees[entree] = imageYd[entree]/255
                potI[neurone] += entrees[entree]*poidsCaches[neurone][entree]

            # print("Poti : " + str(potI[neurone]))
            xH[neurone] = sigmoid(potI[neurone])


        # Partie des neurones finaux (puisque gestion d'erreurs)
        potI = [0] * 100
        for neurone in range(0,10):
            for entree in range(0,100):
                entrees[entree] = xH[entree]
                potI[neurone] += entrees[entree]*poidsFinaux[neurone][entree]

            print("Poti : " + str(potI[neurone]))
            xI[neurone] = sigmoid(potI[neurone])

            # print("Result : " + str(xI))
            deltaFinaux[neurone] = (sigmoid(potI[neurone]) * (1 - sigmoid(potI[neurone]))) * yd[neurone] - xI[neurone]
            print(sigmoid(potI[neurone]))
            print(1 - sigmoid(potI[neurone]))
            print(yd[neurone])
            print(xI[neurone])
            print(deltaFinaux[neurone])
            print("")

            for entree in range(0, 100):
                sigmaFinaux[neurone] += deltaFinaux[neurone] * poidsFinaux[neurone][entree]

        print(yd)
        print(xI)

        for neuroneCaches in range(0, 100):
            for neuroneFinaux in range(0, 10):
                deltaCaches[neuroneCaches] = (potI[neuroneCaches] * (1 - potI[neuroneCaches])) * sigmaFinaux[neuroneFinaux]

        for neurone in range(0,10):
            for entree in range(0, 100):
                poidsFinaux[neurone][entree] += epsilon * deltaFinaux[neurone] * xI[neurone]

        for neurone in range(0,100):
            for entree in range(0, 784):
                poidsCaches[neurone][entree] += epsilon * deltaCaches[neurone] * entrees[neurone]


        # Calculer les erreurs
        # De ce que j'ai compris, faire 100 apprentissage -> test -> si % d'erreur > 4% -> re-boucler

        for neuroneFinaux in range(0, 10):
            erreurTotal += deltaFinaux[neuroneFinaux]
        erreurTotal = erreurTotal/p

        listErreurTotale[i] = abs(erreurTotal)
        print("Erreur Totale : " + str(abs(erreurTotal)))
        print("")
    graphiqueErreurApprentissage(listErreurTotale)
    print("Fin apprentissage")

def graphiqueErreurApprentissage(liste):
    indices = list(range(len(liste)))
    # Tracer la courbe
    plt.figure(figsize=(10, 6))
    plt.plot(indices, liste, marker='o', linestyle='-', color='b', label='Nb Erreurs')

    # Personnalisation du graphique
    plt.title("Courbe du Nombre d'Erreurs Totale en Fonction du nombre d'itérations")
    plt.xlabel("Itérations")
    plt.ylabel("Nombre d'Erreurs")
    plt.grid(True)
    plt.legend()

    # Afficher le graphique
    plt.show()

def test():
    p = 1000
    listErreurTotale = [0] * p
    epsilon = 0.01
    for i in range(0, p):
        print(i)
        potI = [0] * 100
        xH = [0] * 100
        xI = [0] * 10
        deltaCaches = [0] * 100
        deltaFinaux = [0] * 10
        sigmaFinaux = [0] * 10

        nbimage = random.randint(0, 9999)
        imageYd, labelYd = testing_data[nbimage]
        entrees = [0] * (28 * 28)  # 28x28

        yd = label_to_one_hot(labelYd)

        for neurone in range(0, 100):
            for entree in range(0, 784):
                entrees[entree] = imageYd[entree] / 255
                potI[neurone] += entrees[entree] * poidsCaches[neurone][entree]

            xH[neurone] = sigmoid(potI[neurone])

        # Partie des neurones finaux (puisque gestion d'erreurs)
        potI = [0] * 100
        for neurone in range(0, 10):
            for entree in range(0, 100):
                entrees[entree] = xH[entree]
                potI[neurone] += entrees[entree] * poidsFinaux[neurone][entree]

            xI[neurone] = sigmoid(potI[neurone])

            # print("Result : " + str(xI))
            deltaFinaux[neurone] = (sigmoid(potI[neurone]) * (1 - sigmoid(potI[neurone]))) * yd[neurone] - xI[neurone]

            for entree in range(0, 100):
                sigmaFinaux[neurone] += deltaFinaux[neurone] * poidsFinaux[neurone][entree]

        for neuroneCaches in range(0, 100):
            for neuroneFinaux in range(0, 10):
                deltaCaches[neuroneCaches] = (potI[neuroneCaches] * (1 - potI[neuroneCaches])) * sigmaFinaux[
                    neuroneFinaux]

        erreurTotal = 0
        indice = 0
        for neuroneFinaux in range(0, 10):
            if deltaFinaux[neuroneFinaux] > erreurTotal:
                erreurTotal = deltaFinaux[neuroneFinaux]
                indice = neuroneFinaux

        print(labelYd)
        print(indice)
        print("")

        if labelYd == indice:
            listErreurTotale[i] = 1

        print("Erreur Totale : " + str(abs(erreurTotal)))
        print("")
    graphiqueErreurApprentissage(listErreurTotale)
    print("Fin apprentissage")

apprentissage()
test()