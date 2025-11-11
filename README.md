# Comptage de Doigts avec Python & OpenCV

[![Version Python](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![Licence: MIT](https://img.shields.io/badge/Licence-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Un projet intelligent de vision par ordinateur qui détecte et compte le nombre de doigts levés en temps réel en utilisant Python, OpenCV et un modèle MobileNet pré-entraîné.

Ce projet démontre comment l'apprentissage profond et la vision par ordinateur peuvent être utilisés pour reconnaître dynamiquement des gestes de la main via une webcam.

## Table des Matières

- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Stack Technique](#stack-technique)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Comment ça Marche](#comment-ça-marche)
- [Structure du Projet](#structure-du-projet)
- [Détails du Modèle](#détails-du-modèle)
- [Améliorations Futures](#améliorations-futures)
- [Dépannage](#dépannage)
- [Contribution](#contribution)
- [Licence](#licence)
- [Auteur](#auteur)

## Aperçu

Le programme utilise OpenCV pour traiter le flux de la caméra, détecter la région de la main et compter les doigts visibles en utilisant des techniques de traitement d'image et de détection d'objets.

Le modèle peut détecter avec précision les mains et classifier les gestes en utilisant une combinaison de :

- MobileNet pré-entraîné (modèle Caffe)
- Fichier de configuration Prototxt
- Inférence en temps réel utilisant le module DNN d'OpenCV

## Fonctionnalités

- Détection de main en temps réel depuis le flux de la webcam
- Comptage automatique des doigts (0–5 doigts)
- Reconnaissance de main basée sur l'apprentissage profond
- Modèle MobileNet SSD léger
- Inférence haute vitesse (environ 30 FPS)
- Facile à utiliser et extensible
- Compatibilité multiplateforme (Windows, macOS, Linux)

## Stack Technique

**Langage de Programmation :**
- Python 3.7+

**Bibliothèques & Frameworks :**
- OpenCV (cv2) - Vision par ordinateur et traitement d'image
- NumPy - Calculs numériques
- imutils - Fonctions de commodité pour OpenCV

**Modèle d'Apprentissage Profond :**
- MobileNet SSD (Caffe)
- Pré-entraîné sur un dataset de détection de mains

## Installation

### Prérequis

Assurez-vous d'avoir Python 3.7 ou supérieur installé sur votre système.

### Étape 1 : Cloner le Dépôt
```bash
git clone https://github.com/omarlr-pro/finger-counts-using-py.git
cd finger-counts-using-py
```

### Étape 2 : Installer les Dépendances
```bash
pip install opencv-python numpy imutils
```

Ou utilisez le fichier requirements :
```bash
pip install -r requirements.txt
```

## Utilisation

### Lancer l'Application

Exécutez simplement le script principal :
```bash
python finger.py
```

La webcam s'ouvrira automatiquement, et le modèle commencera à détecter votre main et à compter les doigts que vous levez en temps réel.

### Contrôles

- Appuyez sur `q` ou `ESC` pour quitter l'application
- Assurez un bon éclairage pour une détection optimale
- Positionnez votre main dans le cadre de la caméra

### Conseils pour de Meilleurs Résultats

- Utilisez un fond uni pour une meilleure détection de la main
- Gardez votre main entre 50 et 150 cm de la caméra
- Assurez un éclairage adéquat dans la pièce
- Écartez clairement vos doigts pour un comptage précis

## Comment ça Marche

Voici une décomposition de la logique derrière le projet :

### 1. Capture d'Images
Le flux de la webcam est capturé en utilisant le module VideoCapture d'OpenCV.

### 2. Détection de la Main
Le modèle MobileNet SSD pré-entraîné est chargé via `deploy.prototxt` et `mobilenet_iter_73000.caffemodel`. Il détecte la boîte englobante de la main dans chaque image.

### 3. Région d'Intérêt (ROI)
La zone de la main est découpée et isolée pour une analyse plus approfondie.

### 4. Logique de Comptage des Doigts

L'algorithme effectue les étapes suivantes :
- Convertit la région de la main en niveaux de gris
- Applique un flou gaussien pour réduire le bruit
- Utilise un seuillage adaptatif pour une meilleure segmentation
- Extrait les contours de la main
- Calcule l'enveloppe convexe et les défauts de convexité
- Détermine le nombre de doigts étendus basé sur une analyse géométrique

### 5. Affichage des Résultats
Le compte actuel des doigts est affiché en temps réel à l'écran avec un retour visuel.

## Structure du Projet
```
finger-counts-using-py/
│
├── finger.py                     # Script principal – exécute la détection et le comptage
├── deploy.prototxt               # Configuration de l'architecture réseau
├── mobilenet_iter_73000.caffemodel  # Poids du modèle pré-entraîné
├── ssd_mobilenet_v2_coco.config  # Fichier de configuration supplémentaire
├── requirements.txt              # Dépendances Python
└── README.md                     # Documentation
```

## Détails du Modèle

| Fichier | Description |
|---------|-------------|
| `deploy.prototxt` | Définit l'architecture du modèle MobileNet |
| `mobilenet_iter_73000.caffemodel` | Contient les poids entraînés pour la détection de main |
| `ssd_mobilenet_v2_coco.config` | Configuration supplémentaire pour l'ajustement fin ou le ré-entraînement |

### Spécifications du Modèle

- **Architecture :** MobileNet SSD (Single Shot Detector)
- **Framework :** Caffe
- **Taille d'Entrée :** 300x300 pixels
- **Sortie :** Boîtes englobantes avec scores de confiance
- **Vitesse d'Inférence :** ~30 FPS sur du matériel moderne

## Améliorations Futures

Fonctionnalités et améliorations potentielles :

- Ajouter la reconnaissance de gestes pour des jeux pierre/papier/ciseaux
- Implémenter l'estimation de pose de la main pour un suivi 3D
- Entraîner un modèle personnalisé avec une précision supérieure
- Contrôler des applications externes ou des robots avec des gestes de la main
- Ajouter le support de détection de plusieurs mains
- Déployer sur appareils mobiles avec TensorFlow Lite
- Créer une interface GUI pour la configuration
- Ajouter une fonctionnalité d'enregistrement pour des datasets de gestes
- Implémenter des commandes basées sur les gestes (contrôle du volume, contrôle de souris)

## Optimisation des Performances

Pour de meilleures performances :
```python
# Réduire la résolution des images
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Sauter des images si nécessaire
frame_skip = 2
```

## Dépannage

**Problème : Caméra non détectée**
```bash
# Tester l'accès à la caméra
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

**Problème : Performance lente**
- Réduire la résolution vidéo
- Fermer les autres applications utilisant la caméra
- Mettre à jour OpenCV vers la dernière version

**Problème : Faible précision de détection**
- Améliorer les conditions d'éclairage
- Utiliser un fond uni
- Assurez-vous que votre main est dans la plage de détection

## Contribution

Les contributions sont les bienvenues ! Voici comment vous pouvez aider :

1. Forkez le projet
2. Créez votre branche de fonctionnalité (`git checkout -b feature/NouvelleFonctionnalité`)
3. Committez vos changements (`git commit -m 'Ajouter une nouvelle fonctionnalité'`)
4. Poussez vers la branche (`git push origin feature/NouvelleFonctionnalité`)
5. Ouvrez une Pull Request

Assurez-vous de mettre à jour les tests si approprié et de respecter le style de code existant.

## Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

Vous êtes libre d'utiliser, modifier et distribuer ce logiciel avec attribution appropriée.

## Auteur

**Omar Laraje**

- GitHub : [@omarlr-pro](https://github.com/omarlr-pro)
- Linkedin : [omar laraje](https://www.linkedin.com/in/omar-laraje/)
- Rôle : Étudiant en Data Science et Buissnes Inteligete
- Localisation : Rabat, Maroc

## Remerciements

- Construit avec Python et OpenCV
- Modèle MobileNet SSD du Caffe Model Zoo
- Inspiré par la recherche en vision par ordinateur et reconnaissance de gestes

## Mots-clés

`Python` · `OpenCV` · `Vision par Ordinateur` · `Apprentissage Profond` · `MobileNet` · `Détection de Main` · `Reconnaissance de Gestes` · `Comptage de Doigts` · `Détection en Temps Réel` · `Traitement d'Image` · `Détection d'Objets` · `SSD` · `Caffe`

---

**Mettez une étoile à ce dépôt si vous l'avez trouvé utile !**

**Fait avec ❤️ par Omar Laraje**
