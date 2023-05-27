import streamlit as st
import numpy as np
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import os
import joblib
import sklearn
import keras.models

#Configuración de la app
st.set_page_config(
    page_title="DigiPredict App -STM-",
    page_icon="🦧",
    layout="centered",
    initial_sidebar_state="expanded",
)

#Parametros
MODEL_NAMES = ("mnist_KerasConv_digits",
            "mnist_KerasConv_digits_nomax",
            "mnist_KerasDense_digits",
            "mnist_RandomForest_sklearn")

MODELS_FOLDER = "models"
IMG_FOLDER = "img"

#st.session_state

#Funciones auxiliares
def ruta_models(model_name):
    return os.path.join(MODELS_FOLDER,model_name)

def ruta_img(model_name):
    return os.path.join(MODELS_FOLDER,IMG_FOLDER,model_name)

def imagen_nula(img):
    #print(np.count_nonzero(img == 0))
    if (np.count_nonzero(img == 0) > 0.95 * img.size) or (np.count_nonzero(img == 0) < 0.10 * img.size):
        st.error("Error: La imagen no es correcta. Es posible que no exista ningún dibujo válido.")
        st.stop()
    
def shape_no_valida(img):
    if img.shape != (28,28):
        st.error(f"Error: La dimensión de la imagen debe ser (28,28) y no {img.shape}")
        st.stop()

def validar_imagen(uploaded_file):
    img = np.array(Image.open(BytesIO(uploaded_file.getvalue())))

    shape_no_valida(img)
    imagen_nula(img)
    
    #Reshepeamos y escalamos.
    #A los modelos de capas dense tambien les alimentamos con esta shape
    
    X = (img.reshape(1,28,28,1).astype(np.float32) - 127.5 ) / (127.5)

    if "sklearn" in model_name:
        X = (img.reshape(1,-1).astype(np.float32) - 127.5 ) / (127.5)

    return X, img

def predecir(digit,model):
    if "sklearn" in model_name:
        return model.predict(digit)[0], round(np.max(model.predict_proba(digit))*100,2)
    return np.argmax(model.predict(digit)),round(max(model.predict(digit)[0])*100,2)
    
@st.cache_resource(show_spinner=False)
def cargar_modelo(model_name):
    """ Carga el modelo desde disco a partir de su nombre """

    if "sklearn" in model_name:
        return joblib.load(ruta_models(model_name) + ".joblib")

    return keras.models.load_model(ruta_models(model_name))

def describir_modelo():
    if model_name == "mnist_KerasConv_digits":
    
        st.caption(
            """ Este modelo está entrenado para distinguir dígitos
            escritos a mano. La Red Neuronal utilizada es de tipo Convolucional.
            Consta de 2 capas MaxPooling después de la primera y la segunda capa Convolucional."""
        )
    elif model_name == "mnist_KerasConv_digits_nomax":
    
        st.caption(
            """ Este modelo está entrenado para distinguir dígitos
            escritos a mano. La Red Neuronal utilizada es de tipo Convolucional.
            Es el mismo modelo que el anterior pero suprimiendo las capas MaxPooling."""
        )
    elif model_name == "mnist_KerasDense_digits":

        st.caption(
            """ Este modelo está entrenado para distinguir dígitos
            escritos a mano. La Red Neuronal utilizada es de tipo 'Dense'.
            4 capas lineales activadas con función no lineal relu."""
        )
    
    elif model_name == "mnist_RandomForest_sklearn":

        st.caption(
            """ Este modelo está entrenado para distinguir dígitos
            escritos a mano. El tipo de clasificador utilizado es de tipo 'Random Forest`
            que consiste en múltiples árboles de decisión entrenados simultáneamente y procesados en paralelo.
            Los parámetros del modelo se ven en el desplegable."""
        )

def init_session():
    if "predicciones" not in st.session_state:
        st.session_state["predicciones"] = {}
        for model in MODEL_NAMES:
            st.session_state["predicciones"][model] = {
                "imagenes":[],
                "predicciones":[],
                "confianzas":[]
            }
            
@st.cache_data(show_spinner=False)
def cargar_matriz_confusion(model_name):
    print(f"{ruta_models(model_name)}.npy")
    return np.load(f"{ruta_models(model_name)}.npy")

@st.cache_data(show_spinner=False)
def plotear_matriz_confusion(conf_matrix):
    import plotly.express as px

    fig = px.imshow(conf_matrix,text_auto=True, aspect=0.05)
    st.plotly_chart(fig, theme="streamlit")

def cargar_imagen_modelo(model_name):
    image = Image.open(f"{ruta_img(model_name)}.png")
    st.image(image,caption="Esquema del modelo")

def actualizar_historico_predicciones(model_name,uploaded_file,pred,conf):
    if uploaded_file.name not in st.session_state["predicciones"][model_name]["imagenes"]:
        st.session_state["predicciones"][model_name]["imagenes"].append(uploaded_file.name)
        st.session_state["predicciones"][model_name]["predicciones"].append(pred)
        st.session_state["predicciones"][model_name]["confianzas"].append(conf)

def plotear_historico_predicciones():
    """ Función para mostrar el dataframe del histórico\n
     de predicciones realizadas por modelo """
    for idx,model in enumerate(MODEL_NAMES,start=1):
        if st.session_state["predicciones"][model].get("imagenes"):
            df_historico = pd.DataFrame(
                st.session_state["predicciones"][model]
            )
            st.write(f":blue[{model}]")
            st.dataframe(df_historico)

#cuerpo principal
if __name__ == '__main__':
    
    init_session()
    #st.session_state["predicciones"]

    st.title(":blue[Digi]Predict App")
    st.markdown("""
     Aplicación para evaluar diferentes modelos entrenados con el dataset [**MNIST**](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) que contiene
                60.000 muestras correspondientes a dígitos del 0 al 9 escritos a mano.
                Los modelos tratan de resolver problemas de *clasificación múltiple supervisada*.
                """)

    model = None
    digit = None

    with st.sidebar:
        st.header(""" Elegir el modelo """)
        model_name = st.selectbox(
            """ Escoge el modelo para las predicciones """,
            MODEL_NAMES,
            index=0,
        )
        
        try:
            with st.spinner("Cargando modelo"):
                model = cargar_modelo(model_name)
                modelo_cargado = True

            with st.expander("Ver modelo"):
                describir_modelo()
                if "Keras" in model_name:
                    st.write("Número de parámetros:",f":green[{model.count_params():,}]")
                    cargar_imagen_modelo(model_name)                

                
                if "sklearn" in model_name:
                    st.write(model.get_params())

                st.subheader("Matriz de confusión")
                try:
                    plotear_matriz_confusion((cargar_matriz_confusion(model_name)))
                except:
                    st.error("Matriz de Confusión no disponible.")

        except Exception as exc:
                print(exc)
                st.error(f"Error al cargar el modelo")
                with st.expander("Ver detalles del error"):
                    st.error(exc)
                modelo_cargado = False


    st.header(""" Cargar una imagen """)
    st.info(""" Las imágenes deben ser de tamaño 28x28 píxeles con 1 solo canal.\n
    El dígito debe estar dibujado en blanco con fondo negro. """)
    uploaded_file = st.file_uploader(
        "Elige una imagen",
        type=["png","tif","jpg","bmp","jpeg"],      
        )

    if uploaded_file is not None:

        digit, img = validar_imagen(uploaded_file)

        if digit is not None:
            st.subheader(""" Información del archivo """)
            st.write("Tipo de archivo:", uploaded_file.type,"|","Dimensiones imagen original:", img.shape,)
            st.write("Dimensiones del input array:",digit.shape)

            with st.expander("Ver imagen"):
                fig, ax = plt.subplots()
                ax.imshow(img,cmap="gray")
                st.pyplot(fig)


    st.divider()
    st.header(""" Lanzar predicción """)

    if digit is None:
        st.info("Es necesario cargar una imagen para predecir.")
        
    else:
        if modelo_cargado:
            if st.button("Lanzar predicción"):
                with st.spinner("Pensando..."):
                    pred, conf = predecir(digit,model)
                
                st.markdown(f"**Modelo**: :blue[*{model_name}*]")
                st.metric("Predicción",pred)
                st.metric("Confianza:",conf)

                actualizar_historico_predicciones(model_name,uploaded_file,pred,conf)
        
        else:
            st.info("Es necesario cargar un modelo para predecir.")

    with st.expander("Ver histórico de predicciones"):
        plotear_historico_predicciones()

            



    


