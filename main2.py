import random
import shutil
import matplotlib.pyplot as plt # A installer si vouloir graphique

fileUn = 'un.txt'
fileZero = 'zero.txt'
fileTaken = ""

poids = []

for i in range(0,48):
    poids.append(random.randint(0,100)/100/48) 

print(poids)
print(len(poids))

def apprentissage():
    i = 0
    listErreurTotale = []
    erreurTotal = 1
    erreurOne = 1
    erreurZero = 0
    teta = 0.5
    epsilon = 0.01
    yd = 0
    while(erreurTotal>0.01):
        potI = 0
        entrees = []
        if i%2 == 0:
            fileTaken = fileZero
        else:
            fileTaken = fileUn

        print("Type file : " + fileTaken)

        with open(fileTaken, 'r') as f:
            for line in f:
                for element in line:
                    if element == ".":
                        entrees.append(0)
                    elif element == "*":
                        entrees.append(1)
                    else:
                        yd = element

        for neurone in range(0,48):
            potI += entrees[neurone]*poids[neurone]

        print("Poti : " + str(potI))
        if potI > teta:
            xI = 1
        else:
            xI = 0

        print("Result : " + str(xI))
        delta = int(yd) - potI

        for neurone in range(len(poids)):
            poids[neurone] += epsilon*delta*entrees[neurone]

        if i%2 == 0:
            erreurZero = delta
        else:
            erreurOne = delta

        erreurTotal = abs(erreurOne) + abs(erreurZero)
        listErreurTotale.append(erreurTotal)
        print("Erreur Totale : " + str(erreurTotal))
        i += 1
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

def createBruit(fichier, pourcentage):
    with open(fichier, 'r') as f:
        contenu = f.read()

    # Convertir le contenu en liste (chaque caractère est modifiable)
    liste_contenu = list(contenu)
    taille = len(liste_contenu)

    # Calculer combien de caractères doivent être changés
    nb_a_changer = int(taille * (pourcentage / 100))

    # Sélectionner des indices aléatoires sans répétition
    indices_a_changer = random.sample(range(taille), nb_a_changer)

    # Modifier les caractères aux indices choisis
    for idx in indices_a_changer:
        if liste_contenu[idx] == '.':
            liste_contenu[idx] = '*'
        elif liste_contenu[idx] == '*':
            liste_contenu[idx] = '.'

    # Convertir la liste modifiée en une chaîne
    contenu_modifie = ''.join(liste_contenu)

    return contenu_modifie

def test(pourcentage, isZero):
    nbErreurBruit = 0
    teta = 0.5
    yd = 0
    for i in range(0,50):
        potI = 0
        entrees = []
        chaineZeroBruit = createBruit(fileZero, pourcentage)
        chaineUnBruit = createBruit(fileUn, pourcentage)
        if isZero:
            chaineTaken = chaineZeroBruit
        else:
            chaineTaken = chaineUnBruit

        for line in chaineTaken:
            for element in line:
                if element == ".":
                    entrees.append(0)
                elif element == "*":
                    entrees.append(1)
                else:
                    yd = element

        for neurone in range(0, 48):
            potI += entrees[neurone] * poids[neurone]

        print("Poti : " + str(potI))
        if potI > teta:
            xI = 1
        else:
            xI = 0

        print("Result : " + str(xI))
        delta = int(yd) - xI

        if delta != 0 :
            nbErreurBruit += 1

        print("")

    print(nbErreurBruit)
    return nbErreurBruit

def graphiqueErreurTest():
    pourcentages = list(range(0, 51, 2))  # De 0% à 50% par pas de 2
    erreursZero = [test(p, True) for p in pourcentages]  # Pour 0
    erreursUn = [test(p, False) for p in pourcentages]  # Pour 1

    # Tracer la courbe
    plt.figure(figsize=(10, 6))
    plt.plot(pourcentages, erreursZero, marker='o', linestyle='-', color='b', label='0')
    plt.plot(pourcentages, erreursUn, marker='s', linestyle='-', color='r', label='1')

    # Personnalisation du graphique
    plt.title("Courbe du Nombre d'Erreurs en Fonction du Pourcentage de bruit")
    plt.xlabel("Pourcentage (%)")
    plt.ylabel("Nombre d'Erreurs")
    plt.grid(True)
    plt.legend()

    # Afficher le graphique
    plt.show()

apprentissage()
# test(50, False)
graphiqueErreurTest()