# DigiPredict App
Aplicación web para cargar y evaluar modelos entrenados con el dataset de dígitos escritos a mano [MNIST](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv).

## Cómo funciona
El funcionamiento es muy sencillo:
- Selecciona un modelo en el menúy de la izquierda. De momento solo hay 3 modelos disponibles (2 convolucionales y 1 de tipo dense).
Puedes ver las características del modelo seleccionado desplegando la pestaña "Ver modelo".
- Carga una imagen mediante el cargador de imagenes. La imagen debe ser una imagen en blanco y negro (1 canal) y de 28 x 28 píxeles.
Puedes ver la imagen que has cargado mediante la pestaña desplegable "Ver imagen".
- Lanza la predicción.

Las predicciones se guardan en un historial que puedes ver desplegando la pestaña de "Ver historial de predicciones".

Ver la app [aquí](https://digi-predict.streamlit.app/).

## Actualizaciones
27/05/2023 - Añadido modelo Random Forest de 500 estimadores.<br>
27/05/2023 - Añadido modelo KNN con 5 neighbors.<br>
27/05/2023 - Añadido modelo Logistic Regression.<br>

