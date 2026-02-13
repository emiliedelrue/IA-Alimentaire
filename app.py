import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os

# Configuration
MODEL_PATH = 'model_v3.h5'
CLASSES_PATH = 'classes.txt'
IMG_SIZE = (150, 150)

# --- Functions ---

@st.cache_resource
def load_food_model():
    """Reconstruit le modèle et charge les poids pour éviter les erreurs de version."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"Modèle introuvable : {MODEL_PATH}")
        return None
    try:
        # Re-création de l'architecture exacte (comme dans FOOD_MODEL_V3.py)
        # Cela contourne les problèmes de compatibilité 'load_model' entre versions Keras
        from tensorflow.keras.applications import ResNet50
        from tensorflow.keras import layers, models
        
        # 1. Base Model
        base_model = ResNet50(weights=None, include_top=False, input_shape=(150, 150, 3))
        base_model.trainable = False # Important : on garde le même état
        
        # 2. Top Layers
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(101, activation='softmax') # 101 classes
        ])
        
        # 3. Chargement des poids
        # Le fichier .h5 contient tout, mais on peut charger juste les poids dans l'architecture
        model.load_weights(MODEL_PATH)
        print("Modèle chargé via reconstruction + load_weights")
        return model
        
    except Exception as e:
        st.error(f"Erreur lors de la reconstruction/chargement : {e}")
        return None

@st.cache_data
def load_classes():
    """Charge la liste des classes depuis le fichier."""
    if not os.path.exists(CLASSES_PATH):
        st.error(f"Fichier de classes introuvable : {CLASSES_PATH}")
        return []
    with open(CLASSES_PATH, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def preprocess_image(image):
    """Prépare l'image pour le modèle (resize + preprocess ResNet)."""
    # Redimensionnement
    img = image.resize(IMG_SIZE)
    # Conversion en array numpy
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    # Ajout de la dimension de batch (1, 150, 150, 3)
    img_array = np.expand_dims(img_array, axis=0)
    # Pré-traitement spécifique à ResNet50 (comme dans l'entraînement)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return img_array

# --- Interface ---

st.set_page_config(page_title="Food Classifier V3")

st.title("Food Classifier V3")
st.write("Téléchargez une photo de plat pour identifier de quoi il s'agit !")

# Chargement des ressources
model = load_food_model()
class_names = load_classes()

uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None and class_names:
    try:
        # Affichage de l'image
        image = Image.open(uploaded_file)
        st.image(image, caption='Image téléchargée', use_container_width=True)
        
        st.write("Analyse en cours...")
        
        # Prédiction
        processed_img = preprocess_image(image)
        predictions = model.predict(processed_img)
        
        # Résultats (Top 3)
        top_k = 3
        top_indices = np.argsort(predictions[0])[::-1][:top_k]
        
        st.success(f"Résultat : **{class_names[top_indices[0]]}** ({predictions[0][top_indices[0]]:.1%})")
        
        st.write("### Détails (Top 3):")
        for i in top_indices:
            class_name = class_names[i]
            confidence = predictions[0][i]
            st.write(f"- **{class_name}**: {confidence:.1%}")
            st.progress(int(confidence * 100))
            
    except Exception as e:
        st.error(f"Une erreur est survenue lors du traitement : {e}")

elif not model:
    st.warning("Le modèle n'a pas pu être chargé.")
elif not class_names:
    st.warning("La liste des classes n'a pas pu être chargée.")
