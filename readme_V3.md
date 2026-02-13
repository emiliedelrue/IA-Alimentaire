# 🍔 FOOD_MODEL_V3 - Classification de Nourriture avec ResNet50 (CPU)

Ce projet implémente un modèle de Deep Learning pour la classification d'images de nourriture en utilisant l'architecture **ResNet50**. Cette version (`V3`) est spécifiquement configurée pour l'entraînement sur **CPU**, sans utiliser de GPU.

## 📋 Description

Le script `FOOD_MODEL_V3.py` effectue les opérations suivantes :
1.  **Force l'utilisation du CPU** : Désactive la détection des GPU pour éviter les erreurs ou les configurations complexes.
2.  **Prépare les données** : Charge les images depuis le dossier `data/food41/images`, les redimensionne et applique le pré-traitement spécifique à ResNet.
3.  **Utilise le Transfer Learning** : Charge un modèle ResNet50 pré-entraîné sur ImageNet (le "cerveau" de base) et gèle ses poids.
4.  **Ajoute une tête de classification** : Ajoute de nouvelles couches pour adapter le modèle aux classes de nourriture spécifiques.
5.  **Entraîne le modèle** : Lance l'entraînement avec sauvegarde automatique à chaque époque (`ModelCheckpoint`) et arrêt précoce (`EarlyStopping`) si le modèle ne progresse plus.
6.  **Sauvegarde** : Enregistre le modèle final sous `food_model_v3_resnet50_cpu.h5`.

## ⚙️ Prérequis

Assurez-vous d'avoir installé les bibliothèques suivantes :

```bash
pip install tensorflow matplotlib numpy
```

## 📂 Structure des Données

Le script s'attend à une structure de dossiers comme suit :

```
projet/
├── FOOD_MODEL_V3.py
├── data/
│   └── food41/
│       └── images/
│           ├── pizza/
│           │   ├── 123.jpg
│           │   └── ...
│           ├── sushi/
│           │   ├── 456.jpg
│           │   └── ...
│           └── ... (autres classes)
└── training_checkpoints_v3/ (créé automatiquement)
```

## 🚀 Utilisation

Pour lancer l'entraînement, exécutez simplement le script avec Python :

```bash
python FOOD_MODEL_V3.py
```

## 🛠️ Configuration

Les paramètres principaux sont définis au début du fichier :

-   `BATCH_SIZE = 32` : Nombre d'images traitées à la fois (réduit pour le CPU).
-   `IMG_HEIGHT = 150`, `IMG_WIDTH = 150` : Dimensions des images.
-   `EPOCHS = 20` : Nombre maximum de cycles d'entraînement.
-   `NUM_CLASSES_TO_USE = 101` : Nombre de classes de nourriture à utiliser (peut être réduit pour tester plus vite).

## 📊 Résultats et Sorties

-   **Checkpoints** : Des modèles `.h5` sont sauvegardés dans `training_checkpoints_v3/` à la fin de chaque époque.
-   **Modèle Final** : Le fichier `food_model_v3_resnet50_cpu.h5` est créé à la fin.
-   **Graphiques** : Une fenêtre s'ouvre à la fin pour afficher les courbes de précision (accuracy) et de perte (loss).
