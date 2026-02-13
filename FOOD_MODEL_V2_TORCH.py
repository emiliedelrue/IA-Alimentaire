import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import pathlib
import time
import copy
import numpy as np

# --- Configuration ---
# Vérification de la disponibilité du GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Appareil utilisé : {device}")

if device.type == 'cuda':
    print(f"Nom du GPU : {torch.cuda.get_device_name(0)}")
    print(f"Mémoire VRAM disponible : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
# Activation de l'optimisation des performances CUDA
    torch.backends.cudnn.benchmark = True
    # Précision mixte pour accélérer l'entraînement sur RTX 4060 Ti
    scaler = torch.amp.GradScaler('cuda')

DATA_DIR = pathlib.Path("data/food41/images")
BATCH_SIZE = 64
IMG_HEIGHT = 224
IMG_WIDTH = 224
EPOCHS = 20
NUM_CLASSES_TO_USE = 101
FINE_TUNE_EPOCHS = 20

class TransformedDataset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
    def __len__(self):
        return len(self.subset)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, scheduler=None, num_epochs=25, is_inception=False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Listes pour stocker les métriques pour les graphiques
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)

        # Chaque époque a une phase d'entraînement et de validation
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Mode entraînement
            else:
                model.eval()   # Mode évaluation

            running_loss = 0.0
            running_corrects = 0

            # Itération sur les données
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Remise à zéro des gradients
                optimizer.zero_grad()

                # Forward
                # Suivi de l'historique seulement en phase d'entraînement
                with torch.set_grad_enabled(phase == 'train'):
                    # Précision mixte (AMP)
                    if device.type == 'cuda' and phase == 'train':
                        with torch.amp.autocast('cuda'):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)
                    
                    _, preds = torch.max(outputs, 1)

                    # Backward + Optimize seulement en phase d'entraînement
                    if phase == 'train':
                        if device.type == 'cuda':
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()

                # Statistiques
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()
            
            if phase == 'train' and scheduler:
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Stockage des métriques
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

            # Copie profonde du modèle si c'est le meilleur
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), 'best_model_checkpoint.pth') # Sauvegarde intermédiaire

        print()

    time_elapsed = time.time() - since
    print(f'Entraînement terminé en {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Meilleure précision val: {best_acc:4f}')

    # Chargement des meilleurs poids
    model.load_state_dict(best_model_wts)
    return model, history

def main():
    # --- Préparation des Données ---
    print(f"Sélection des classes...")
    # Transformations des données
    # ResNet attend des images normalisées avec mean=[0.485, 0.456, 0.406] et std=[0.229, 0.224, 0.225]
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(IMG_HEIGHT), # Augmentation des données
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(IMG_HEIGHT),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # IMPORTANT: On suppose que DATA_DIR contient des sous-dossiers pour chaque classe
    # On filtre pour ne garder que les NUM_CLASSES_TO_USE premières classes
    # Comme ImageFolder charge tout, on va devoir être astucieux ou simplement charger tout et filtrer si possible, 
    # mais ImageFolder standard scanne tout.
    # Pour simplifier et respecter la structure existante qui filtrait manuellement, on va utiliser ImageFolder
    # mais on va limiter le dataset si besoin en masquant les dossiers, ou simplement accepter tout le dataset si le dossier ne contient QUE food41.
    # Si le dataset est complet (101 classes), ImageFolder chargera tout. Pour limiter à NUM_CLASSES_TO_USE,
    # on peut manipuler dataset.samples ou dataset.classes, mais le plus propre est de déplacer les dossiers non voulus, ce qu'on ne veut pas faire.
    # Alternative : Créer un CustomDataset.
    
    # Pour cet exemple et pour garantir la robustesse, on va utiliser une approche simple :
    # Si le dossier contient 101 classes, on charge tout. Si on veut limiter, on le ferait ici.
    # Le script TF d'origine listait les dossiers. On va faire similaire avec un Custom Dataset ou Subset.
    
    # Pour faire simple et rapide avec PyTorch : on utilise ImageFolder sur tout le répertoire
    # Mais on réduit le dataset via 'Subset' si on veut vraiment limiter (ce qui n'économise pas la RAM de chargement des classes mais réduit le temps d'epoch).
    full_dataset = datasets.ImageFolder(DATA_DIR)
    
    # Filtrage des classes pour correspondre à la logique "NUM_CLASSES_TO_USE"
    classes = full_dataset.classes
    if len(classes) > NUM_CLASSES_TO_USE:
        print(f"Attention: Le dossier contient {len(classes)} classes, mais on en utilise que {NUM_CLASSES_TO_USE}.")
        # On sélectionne les indices correspondant aux 101 premières classes
        # Note: Cela suppose que ImageFolder trie les classes par ordre alphabétique, ce qui est le cas.
        selected_indices = [i for i, (path, label) in enumerate(full_dataset.samples) if label < NUM_CLASSES_TO_USE]
        # Recréer un dataset partiel
        dataset = torch.utils.data.Subset(full_dataset, selected_indices)
        # Hack pour que le subset ait l'attribut classes (pour l'affichage plus tard si besoin)
        dataset.classes = classes[:NUM_CLASSES_TO_USE]
    else:
        dataset = full_dataset

    # Séparation Train/Val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Application des transformations
    # Attention: random_split ne permet pas d'avoir des transforms différents facilement car ils partagent le dataset parent.
    # Solution courante : Créer une classe Helper ou ré-instancier. 
    # Ici, pour faire simple et efficace : on applique les transforms au moment du chargement si on avait deux ImageFolder séparés.
    # Comme on divise un seul dataset, on va wrapper.
    
    train_dataset = TransformedDataset(train_dataset, transform=data_transforms['train'])
    val_dataset = TransformedDataset(val_dataset, transform=data_transforms['val'])

    image_datasets = {'train': train_dataset, 'val': val_dataset}
    
    # DataLoaders
    # num_workers=4 est standard, mais sur Windows, il faut être dans if __name__ == '__main__':
    # pin_memory=True accélère le transfert vers le GPU
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                 shuffle=True if x == 'train' else False, num_workers=4, pin_memory=True)
                   for x in ['train', 'val']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = getattr(dataset, 'classes', full_dataset.classes[:NUM_CLASSES_TO_USE])
    
    print(f"Classes : {dataset_sizes}")
    
    # --- Modèle ---
    print("Chargement de ResNet50 (pre-trained)...")
    # Weights='DEFAULT' charge les meilleurs poids disponibles (ImageNet)
    model_ft = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

    # Phase 1 : Geler les paramètres du backbone
    for param in model_ft.parameters():
        param.requires_grad = False

    # Modification de la dernière couche (la tête)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, NUM_CLASSES_TO_USE) # Pas de Softmax ici, CrossEntropyLoss le fait
    )

    model_ft = model_ft.to(device)

    # Fonction de perte
    criterion = nn.CrossEntropyLoss()

    # Optimiseur pour la phase 1 (seulement la tête)
    optimizer_ft = optim.Adam(model_ft.fc.parameters(), lr=0.001)

    # --- Entraînement Phase 1 ---
    print("Démarrage de la Phase 1 : Entraînement des nouvelles couches...")
    model_ft, history_phase1 = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, num_epochs=EPOCHS)

    # --- Phase 2 : Fine-Tuning ---
    print("\n--- Démarrage de la Phase 2 : Fine-Tuning ---")
    
    # Débloquer toutes les couches
    for param in model_ft.parameters():
        param.requires_grad = True

    # Nouvel optimiseur avec Learning Rate très bas pour tout le modèle
    optimizer_fine = optim.Adam(model_ft.parameters(), lr=1e-5)

    # Entraînement Phase 2
    model_ft, history_phase2 = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_fine, num_epochs=FINE_TUNE_EPOCHS)

    # --- Sauvegarde ---
    torch.save(model_ft.state_dict(), 'food_model_v2_resnet50_finetuned.pth')
    print("Modèle final sauvegardé sous : food_model_v2_resnet50_finetuned.pth")

    # --- Visualisation ---
    # Combinaison des historiques
    acc = history_phase1['train_acc'] + history_phase2['train_acc']
    val_acc = history_phase1['val_acc'] + history_phase2['val_acc']
    loss = history_phase1['train_loss'] + history_phase2['train_loss']
    val_loss = history_phase1['val_loss'] + history_phase2['val_loss']

    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label="Précision d'entraînement")
    plt.plot(epochs_range, val_acc, label='Précision de validation')
    plt.plot([EPOCHS-1, EPOCHS-1], plt.ylim(), label='Début Fine-Tuning', linestyle='--')
    plt.legend(loc='lower right')
    plt.title("Précision d'entraînement et de validation")

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label="Perte d'entraînement")
    plt.plot(epochs_range, val_loss, label='Perte de validation')
    plt.plot([EPOCHS-1, EPOCHS-1], plt.ylim(), label='Début Fine-Tuning', linestyle='--')
    plt.legend(loc='upper right')
    plt.title("Perte d'entraînement et de validation")

    plt.show()

if __name__ == '__main__':
    # Fix pour Windows lors de l'utilisation de multiprocessing (DataLoaders)
    main()
