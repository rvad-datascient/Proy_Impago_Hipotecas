# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 🧱 Módulos personalizados
from limpieza_basica import LimpiezaBasica
from preprocesamiento import winsorizar
from sklearn.preprocessing import RobustScaler

# 🌐 Configuración inicial
st.set_page_config(page_title="Predicción de Impagos Hipotecarios", layout="centered")
st.title("🔍 Predicción de Impagos Hipotecarios")

# 📦 Cargar modelo y escalador
try:
    modelo = joblib.load("05_ml_project/modelo_voting_fixed.pkl")
    scaler = joblib.load("05_ml_project/scaler_robust.pkl")

except FileNotFoundError:
    st.error("❌ Archivos 'modelo_voting.pkl' o 'scaler_robust.pkl' no encontrados en el directorio.")
    st.stop()

# 📥 Función para obtener inputs del usuario
st.sidebar.header("📥 Datos del cliente")

def input_usuario():
    return pd.DataFrame([{
        'RevolvingUtilizationOfUnsecuredLines': st.sidebar.slider("Uso del Crédito (%)", 0.0, 2.0, 0.5),
        'age': st.sidebar.slider("Edad", 18, 100, 35),
        'NumberOfTime30-59DaysPastDueNotWorse': st.sidebar.number_input("Pagos 30-59 días tarde", 0, 10, 0),
        'NumberOfTime60-89DaysPastDueNotWorse': st.sidebar.number_input("Pagos 60-89 días tarde", 0, 10, 0),
        'NumberOfTimes90DaysLate': st.sidebar.number_input("Pagos 90+ días tarde", 0, 10, 0),
        'DebtRatio': st.sidebar.slider("Ratio de Deuda", 0.0, 3.0, 0.4),
        'MonthlyIncome': st.sidebar.number_input("Ingreso mensual (€)", 0, 50000, 2500),
        'NumberOfOpenCreditLinesAndLoans': st.sidebar.number_input("Líneas de crédito abiertas", 0, 20, 5),
        'NumberRealEstateLoansOrLines': st.sidebar.number_input("Préstamos hipotecarios activos", 0, 5, 1),
        'NumberOfDependents': st.sidebar.number_input("Número de dependientes", 0, 10, 1)
    }])

# 🧼 Función para transformar los datos ingresados
def transformar_datos(input_df):
    limpiador = LimpiezaBasica()
    input_limpio = limpiador.fit_transform(input_df)

    columnas_winsor = [
        'Edad', 'UsoCrédito', 'IngresoMensual', 'RatioDeuda',
        '30-59DíasTarde', '60-89DíasTarde', '90DíasTarde',
        'LíneasCrédito', 'PréstamosCasa', 'Dependientes'
    ]

    input_winsor = winsorizar(input_limpio, columnas_winsor)
    input_scaled = scaler.transform(input_winsor)
    return input_scaled, input_limpio

# 🚀 Ejecutar predicción
input_df = input_usuario()

try:
    input_procesado, input_limpio = transformar_datos(input_df)
    prob = modelo.predict_proba(input_procesado)[0][1]
    umbral = 0.57
    pred = int(prob >= umbral)
except Exception as e:
    st.error(f"❌ Error al procesar los datos: {e}")
    st.stop()

# 📋 Mostrar los datos procesados
st.subheader("📋 Datos del cliente (ya transformados)")
st.write(input_limpio)

# 📊 Mostrar resultado de la predicción
st.subheader("📊 Resultado de la predicción")
if pred == 1:
    st.error(f"⚠️ Riesgo de impago detectado (probabilidad: {prob:.2%})")
else:
    st.success(f"✅ No se detecta riesgo de impago (probabilidad: {prob:.2%})")

# 📈 Barra de progreso
st.write("Probabilidad de impago:")
st.progress(min(int(prob * 100), 100))
st.caption(f"🔎 Umbral de decisión usado: {umbral}")
