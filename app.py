import streamlit as st
import pandas as pd
import joblib
from datetime import datetime
import numpy as np

# Cargar el modelo y el pipeline de preprocesamiento
try:
    # Asegúrate de que los archivos .joblib estén en el mismo directorio que app.py
    model = joblib.load('modelo_churn.joblib')
    preprocessor = joblib.load('pipeline_preproc.joblib')
except FileNotFoundError:
    st.error("Error: Archivos 'modelo_churn.joblib' o 'pipeline_preproc.joblib' no encontrados. Asegúrate de que estén en el mismo directorio que app.py o proporciona la ruta completa.")
    st.stop()

# Título de la aplicación
st.title('Sistema de Alerta Temprana de Churn - Eco-Ride')

st.write("Ingrese los datos del cliente para analizar el riesgo de cancelación.")

# Controles web interactivos para ingresar datos del cliente
with st.form("churn_prediction_form"):
    st.subheader("Datos del Cliente")

    edad = st.slider('Edad', min_value=18, max_value=90, value=30)
    plan_options = ['Básico', 'Premium', 'Elite']
    plan = st.selectbox('Plan', options=plan_options)
    # Rangos ajustados basados en el análisis exploratorio de datos
    uso_mensual_km = st.number_input('Uso Mensual Km', min_value=0.0, max_value=200.0, value=75.0, step=1.0)
    soporte_tickets = st.slider('Soporte Tickets', min_value=0, max_value=30, value=1)
    gasto_promedio = st.number_input('Gasto Promedio', min_value=0.0, max_value=250.0, value=50.0, step=0.1)
    region_options = ['Norte', 'Sur', 'Centro']
    region = st.selectbox('Región', options=region_options)

    # Botón de acción
    submitted = st.form_submit_button("Analizar Riesgo")

    if submitted:
        # Lógica interna: Preparar los datos de entrada en un DataFrame
        # La columna 'Dias_Antiguedad' se calculó en el entrenamiento.
        # Para un nuevo cliente sin 'Fecha_Registro', asumimos que se acaba de registrar,
        # resultando en 'Dias_Antiguedad' igual a 0.
        dias_antiguedad = 0 # Asumiendo que el cliente es "nuevo" y se registra hoy.

        # Crear el DataFrame de entrada con el orden y nombres de columnas exactos de X_train
        # Asegurarse de que 'Plan' esté en minúsculas como se preprocesó durante el entrenamiento.
        input_data = pd.DataFrame([[ 
            float(edad), # Asegurar el tipo float para consistencia con 'Edad' en el modelo
            plan.lower(),
            float(uso_mensual_km),
            int(soporte_tickets),
            float(gasto_promedio),
            region,
            int(dias_antiguedad)
        ]], columns=['Edad', 'Plan', 'Uso_Mensual_Km', 'Soporte_Tickets', 'Gasto_Promedio', 'Region', 'Dias_Antiguedad'])

        # Aplicar la transformación usando el pipeline cargado
        try:
            processed_input = preprocessor.transform(input_data)
        except Exception as e:
            st.error(f"Error durante el preprocesamiento de los datos. Por favor, verifique los valores ingresados. Detalles: {e}")
            st.stop()

        # Realizar la inferencia
        prediction = model.predict(processed_input)
        prediction_proba = model.predict_proba(processed_input)

        st.subheader("Resultado del Análisis:")

        if prediction[0] == 1:
            st.error(f"**🔴 Alto Riesgo de Cancelación**")
            st.write(f"Probabilidad de Cancelación: **{prediction_proba[0][1]*100:.2f}%**")
        else:
            st.success(f"**🟢 Cliente Estable**")
            st.write(f"Probabilidad de Estabilidad: **{prediction_proba[0][0]*100:.2f}%**")
            st.write(f"Probabilidad de Cancelación: **{prediction_proba[0][1]*100:.2f}%**") # Mostrar ambas probabilidades para mayor transparencia
