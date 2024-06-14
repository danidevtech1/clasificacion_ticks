import cv2
import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image

# Cargar el modelo
model = load_model('tick_classification_model.h5')

# Etiquetas
labels = {0: 'larva', 1: 'ninfa', 2: 'adult'}

# Predecimos la clase de la imagen
def predict_stage(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error al leer la imagen de la ruta: {image_path}")
    
    img = cv2.resize(img, (150, 150))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    
    # Predecimos la clase
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    return labels[class_idx], predictions[0]

# Configuración de Streamlit
st.sidebar.image("logo.jpg", use_column_width=True)
st.sidebar.write("Este proyecto se centra en la clasificación de las etapas de desarrollo de las garrapatas utilizando aprendizaje automático y visión por computadora. Las etapas incluyen larva, ninfa y adulto, cada una con características distintas.")
st.sidebar.title("Datos Personales")
st.sidebar.write("Nombre: Cristian Daniel Ccopa Acero")
st.sidebar.write("Universidad: Universidad Nacional del Altiplano")
st.sidebar.write("Carrera: Ingeniería Estaditisca e Informática")
st.sidebar.write("Correo: danielccopa76@gmail.com")
st.sidebar.write("Teléfono: +51 916 330 154")

st.title('TICKSTAGE CLASSIFIER')
st.title('Clasificador de etapas de garrapatas')

uploaded_file = st.file_uploader("Elige la imagen...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Clasificando...")
    
    # Guardar la imagen subida
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Predecir la clase
    try:
        stage, probabilities = predict_stage("temp.jpg")
        st.write(f"## El animal está en la etapa: {stage.upper()} ")
        
        # Mostrar las probabilidades
        st.write("### Probabilidades:")
        st.write(f"Larva: {probabilities[0]:.2%}")
        st.write(f"Ninfa: {probabilities[1]:-.2%}")
        st.write(f"Adulto: {probabilities[2]:.2%}")
    except Exception as e:
        st.write(f"Error: {e}")

st.write("")
st.write("## Descripción de las Etapas de la Garrapata")

col1, col2, col3 = st.columns(3)

with col1:
    st.image("larva.jpg", use_column_width=True)
    st.write("### Larva")
    st.write("Las larvas de garrapata son las primeras etapas del ciclo de vida de la garrapata. Son pequeñas y tienen seis patas.")

with col2:
    st.image("ninfa.jpg", use_column_width=True)
    st.write("### Ninfa")
    st.write("Las ninfas de garrapata son la segunda etapa del ciclo de vida. Son más grandes que las larvas y tienen ocho patas.")

with col3:
    st.image("adulto.jpg", use_column_width=True)
    st.write("### Adulto")
    st.write("Las garrapatas adultas son las etapas finales del ciclo de vida. Son las más grandes y son capaces de reproducirse.")

st.write("")
st.write("Desarrollado por Cristian Ccopa, Universidad Nacional del Altiplano, Ingeniería Estadística e Informática.")
