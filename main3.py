import random
import matplotlib.pyplot as plt

# Fichiers des chiffres de 0 à 9
fichiers = ["zero.txt", "un.txt", "deux.txt",
            "trois.txt", "quatre.txt", "cinq.txt",
            "six.txt", "sept.txt", "huit.txt", "neuf.txt"]

poids = []
for i in range(10):  # 10 neurones de sortie
    ligne_poids = []
    for j in range(48):  # 48 entrées
        ligne_poids.append(random.uniform(-1, 1) / 48)
    poids.append(ligne_poids)


def apprentissage():
    i = 0
    erreurs_totales = []
    erreur_total = 1
    teta = 0.5
    epsilon = 0.01

    while erreur_total > 0.01:
        # Fichier chiffre aléatoire
        fichier_choisi = random.choice(fichiers)
        yd = [0] * 10 # Init sorties

        entrees = []
        with open(fichier_choisi, 'r') as f:
            for line in f:
                for element in line:
                    if element == ".":
                        entrees.append(0)
                    elif element == "*":
                        entrees.append(1)
                    elif element.isdigit():
                        yd[int(element)] = 1

        sorties = []
        for j in range(10):
            potI = 0
            for k in range(48):
                potI += entrees[k] * poids[j][k]
            sorties.append(potI)

        xI = []
        for s in sorties:
            if s > teta:
                xI.append(1)
            else:
                xI.append(0)

        # Erreurs
        deltas = []
        for j in range(10):
            deltas.append(yd[j] - sorties[j])

        # MAJ des poids
        for j in range(10):
            for k in range(48):
                poids[j][k] += epsilon * deltas[j] * entrees[k]

        # Erreur totale
        erreur_total = sum(abs(d) for d in deltas)
        erreurs_totales.append(erreur_total)
        print(f"Itération {i}, Erreur Totale: {erreur_total}")
        i += 1

    graphique_erreur_apprentissage(erreurs_totales)


def graphique_erreur_apprentissage(erreurs):
    indices = list(range(len(erreurs)))
    # Tracer la courbe
    plt.figure(figsize=(10, 6))
    plt.plot(indices, erreurs, marker='o', linestyle='-', color='b', label='Nb Erreurs')

    # Personnalisation du graphique
    plt.title("Courbe du Nombre d'Erreurs Totale en Fonction du nombre d'itérations")
    plt.xlabel("Itérations")
    plt.ylabel("Nombre d'Erreurs")
    plt.grid(True)
    plt.legend()
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

def test(pourcentage):
    erreurs_bruit = 0
    teta = 0.5

    for i in range(100):
        entrees = []
        fichier_choisi = random.choice(fichiers)
        fichier = createBruit(fichier_choisi, pourcentage)
        yd = [0] * 10

        for line in fichier:
            for element in line:
                if element == ".":
                    entrees.append(0)
                elif element == "*":
                    entrees.append(1)
                elif element.isdigit():
                    yd[int(element)] = 1

        sorties = []
        for j in range(10):
            potI = 0
            for k in range(48):
                potI += entrees[k] * poids[j][k]
            sorties.append(potI)

        xI = []
        for s in sorties:
            if s > teta:
                xI.append(1)
            else:
                xI.append(0)

        if xI != yd:
            erreurs_bruit += 1

    return erreurs_bruit


def graphique_erreur_test():
    pourcentages = list(range(0, 51, 1))  # De 0 à 50% de bruit
    erreurs = []
    for p in pourcentages:
        erreurs.append(test(p))

    plt.plot(pourcentages, erreurs, marker='o', linestyle='-', color='r', label='Erreurs')
    plt.title("Nombre d'Erreurs en Fonction du Pourcentage de Bruit")
    plt.xlabel("Pourcentage de bruit (%)")
    plt.ylabel("Nombre d'Erreurs")
    plt.grid(True)
    plt.legend()
    plt.show()


# Exécution
apprentissage()
graphique_erreur_test()