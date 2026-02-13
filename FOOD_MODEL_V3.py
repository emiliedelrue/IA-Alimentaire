import os
# --- Configuration CPU Force ---
# Désactive la visibilité des GPU pour forcer l'utilisation du CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import pathlib
import numpy as np

# --- Configuration Générale ---
DATA_DIR = pathlib.Path("data/food41/images")
BATCH_SIZE = 32          # Réduit à 32 car le CPU est plus lent et a moins de parallélisme que le GPU
IMG_HEIGHT = 150
IMG_WIDTH = 150
EPOCHS = 20
NUM_CLASSES_TO_USE = 101
AUTOTUNE = tf.data.AUTOTUNE

print(f"Version TensorFlow : {tf.__version__}")
# Vérification que le GPU n'est PAS détecté
physical_devices = tf.config.list_physical_devices('GPU')
print(f"Nb de GPUs disponibles (doit être 0) : {len(physical_devices)}")

# --- Chargement des données ---
print(f"Sélection des {NUM_CLASSES_TO_USE} premières classes...")

all_classes = sorted([d.name for d in DATA_DIR.iterdir() if d.is_dir()])
selected_classes = all_classes[:NUM_CLASSES_TO_USE]
print(f"Classes sélectionnées : {selected_classes}")

# Collecte des chemins de fichiers et des étiquettes
file_paths = []
labels = []
class_to_idx = {cls: i for i, cls in enumerate(selected_classes)}

for cls in selected_classes:
    cls_dir = DATA_DIR / cls
    for img_path in cls_dir.glob("*.jpg"):
        file_paths.append(str(img_path))
        labels.append(class_to_idx[cls])

print(f"Total d'images : {len(file_paths)}")

# Création du Dataset
def process_path(file_path, label):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH])
    # Important : Utilisation du pré-traitement spécifique à ResNet50
    img = preprocess_input(img)
    return img, label

ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
ds = ds.shuffle(buffer_size=len(file_paths), seed=123)
ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)

# Séparation Entraînement / Validation
val_size = int(len(file_paths) * 0.2)
train_ds = ds.skip(val_size)
val_ds = ds.take(val_size)

# Mise en lots (Batching) et pré-chargement (Prefetching)
train_ds = train_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)

# --- Architecture du Modèle (Transfer Learning avec ResNet50) ---
# Chargement du modèle ResNet50 sans les couches de classification (include_top=False)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# On gèle les poids du modèle de base (Transfer Learning pur)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES_TO_USE, activation='softmax') # float32 par défaut (pas de mixed precision)
])

model.summary()

# --- Compilation ---
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

# --- Rappels (Callbacks) ---
# Modification V3 : Sauvegarde du modèle complet (.keras ou .h5) à chaque époque
checkpoint_path = "training_checkpoints_v3/model_epoch_{epoch:02d}.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Callback pour sauvegarder le modèle complet à chaque étape (chaque époque)
cp_callback = callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=False, # Sauvegarde tout le modèle (architecture + poids + config)
    save_freq='epoch'        # Sauvegarde à chaque fin d'époque
)

early_stopping = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# --- Phase Unique : Entraînement (Pas de Fine-Tuning) ---
print("Démarrage de l'entraînement sur CPU...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[cp_callback, early_stopping]
)

# --- Visualisation ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Précision d'entraînement")
plt.plot(epochs_range, val_acc, label='Précision de validation')
plt.legend(loc='lower right')
plt.title("Précision d'entraînement et de validation")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Perte d'entraînement")
plt.plot(epochs_range, val_loss, label='Perte de validation')
plt.legend(loc='upper right')
plt.title("Perte d'entraînement et de validation")

plt.show()

# Sauvegarde du modèle final
final_model_path = 'food_model_v3_resnet50_cpu.h5'
model.save(final_model_path)
print(f"Modèle final sauvegardé sous : {final_model_path}")
