import streamlit as st
import pandas as pd
import joblib
import os
import sys
import importlib
import plotly.express as px

# NO silenciamos warnings. Queremos ver si algo falla para arreglarlo de raíz.

# --- CONFIGURACIÓN DE RUTAS RELATIVAS ---
current_dir = os.path.dirname(__file__)
root_dir = os.path.join(current_dir, "..")
if root_dir not in sys.path:
    sys.path.append(root_dir)

# --- PALETA DE COLORES CORPORATIVA (Notebook 02) ---
PALETA = ['#FF6200', '#666666', '#E01E5A'] 

# --- CARGA DINÁMICA DEL MÓDULO DE LIMPIEZA ---
try:
    # Importación explícita para evitar problemas de path
    limpieza_mod = importlib.import_module("03_src.limpieza")
    LimpiezaBasica = limpieza_mod.LimpiezaBasica
except Exception as e:
    st.error(f"Error crítico: No se pudo cargar '03_src/limpieza.py'. Detalle: {e}")
    st.stop()

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Terminal Bancaria - Hipotecas", layout="wide", page_icon="🏦")

@st.cache_resource
def load_models():
    p = os.path.join(root_dir, "04_models")
    # Cargamos archivos usando rutas construidas de forma segura
    return (
        joblib.load(os.path.join(p, 'limpiador_v1.joblib')),
        joblib.load(os.path.join(p, 'scaler_standard_v1.joblib')),
        joblib.load(os.path.join(p, 'credit_scorecard_v1_G71.joblib')),
        joblib.load(os.path.join(p, 'model_features_v1.joblib'))
    )

try:
    limpiador, scaler, modelo, model_features = load_models()
except Exception as e:
    st.error(f"Error al cargar los modelos (.joblib): {e}")
    st.info("Asegúrate de que los archivos estén en la carpeta 04_models")
    st.stop()

def clasificar_cliente(score, uso_credito, debt_ratio):
    """Lógica de negocio para determinar la viabilidad del préstamo"""
    if uso_credito > 0.9: 
        motivo = "Exceso de uso en tarjetas"
    elif debt_ratio > 0.6: 
        motivo = "Sobreendeudamiento"
    else: 
        motivo = "Riesgo por perfil histórico"

    if score < 0.30: return "Pre-concedido (Campaña)", "N/A", "🟢"
    elif score < 0.50: return "Requiere Aval / Garantía", "Reforzar solvencia", "🟡"
    else: return "Denegado", motivo, "🔴"

# --- INTERFAZ PRINCIPAL ---
st.title("🏦 Terminal de Decisiones Crediticias")
st.write("---")

menu = st.sidebar.radio(
    "Menú de Navegación", 
    ["📊 Dashboard Team Leader", "🔍 Buscador de Clientes", "📝 Simulador de Riesgo"]
)

# Ruta al dataset generado en el notebook 02
path_csv = os.path.join(root_dir, "01_data", "processed", "scoring_output.csv")

if menu == "📊 Dashboard Team Leader":
    st.subheader("Estado Global de la Cartera")
    try:
        df_p = pd.read_csv(path_csv)
        total_solicitudes = len(df_p)
        
        # Reindexamos para asegurar que el orden y los colores coincidan
        orden_decisiones = ["Pre-concedido (Campaña)", "Requiere Aval / Garantía", "Denegado"]
        conteo = df_p["decision"].value_counts().reindex(orden_decisiones).fillna(0)
        pct = (conteo / total_solicitudes) * 100

        # Métricas principales usando .iloc para evitar FutureWarning de Pandas
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Solicitudes", f"{total_solicitudes:,}")
        m2.metric("Aprobado", f"{pct.iloc[0]:.1f}%", f"{int(conteo.iloc[0]):,} exp.")
        m3.metric("A Revisar", f"{pct.iloc[1]:.1f}%", f"{int(conteo.iloc[1]):,} exp.")
        m4.metric("Denegado", f"{pct.iloc[2]:.1f}%", f"{int(conteo.iloc[2]):,} exp.")

        st.markdown("### 👥 Planificación y Reparto de Staff")
        exp_por_gestor = 60
        gestores_necesarios = round(total_solicitudes / exp_por_gestor)
        st.success(f"Gestores necesarios para esta cartera: **{gestores_necesarios:,} especialistas**.")

        # Gráfico de barras usando Plotly para interactividad
        df_plot = pd.DataFrame({"Estado": conteo.index, "Expedientes": conteo.values})
        fig = px.bar(df_plot, x="Estado", y="Expedientes", color="Estado",
                     color_discrete_sequence=PALETA, text_auto=',.0f', 
                     title="PLAN DE ACCIÓN OPERATIVO")
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Clientes")
        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.error(f"No se encontró el archivo: {path_csv}. Ejecuta primero el notebook de inferencia.")

elif menu == "🔍 Buscador de Clientes":
    st.subheader("Consulta Individual de Expediente")
    try:
        df_p = pd.read_csv(path_csv)
        id_cliente = st.number_input("ID de Cliente (Índice):", 0, len(df_p)-1, 0)
        
        # Acceso seguro por posición
        cliente = df_p.iloc[id_cliente]
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 📄 Datos del Solicitante")
            st.write(f"**Edad:** {cliente.get('Edad', 'N/A')}")
            st.write(f"**Ingreso Mensual:** ${cliente.get('IngresoMensual', 0):,.2f}")
            st.write(f"**Uso Crédito:** {cliente.get('UsoCrédito', 0):.2%}")
            st.write(f"**Ratio Deuda:** {cliente.get('RatioDeuda', 0):.2%}")
            st.write(f"**Dependientes:** {int(cliente.get('Dependientes', 0))}")
        with c2:
            st.markdown("#### ⚖️ Dictamen del Modelo")
            dec = cliente['decision']
            color = "green" if "Pre-concedido" in dec else "orange" if "Requiere" in dec else "red"
            st.markdown(f"### :{color}[{dec}]")
            st.write(f"**Score de Riesgo:** `{cliente['score']:.4f}`")
            st.write(f"**Motivo Principal:** {cliente['motivo_principal']}")
    except Exception as e:
        st.error(f"Error al consultar el registro: {e}")

elif menu == "📝 Simulador de Riesgo":
    st.subheader("Evaluación de Nueva Solicitud (Inferencia)")
    with st.form("sim_form"):
        c1, c2 = st.columns(2)
        with c1:
            edad = st.number_input("Edad", 18, 100, 35)
            ing = st.number_input("Ingresos Mensuales ($)", 0, 100000, 3000)
            uso = st.slider("Uso Crédito (%)", 0.0, 2.0, 0.3)
            dep = st.number_input("Dependientes", 0, 15, 0)
        with c2:
            deu = st.slider("Ratio Deuda", 0.0, 2.0, 0.3)
            r30 = st.number_input("Retrasos 30-59 días", 0, 50, 0)
            r60 = st.number_input("Retrasos 60-89 días", 0, 50, 0)
            r90 = st.number_input("Retrasos +90 días", 0, 50, 0)
        
        if st.form_submit_button("Realizar Scoring"):
            # Construimos el DataFrame con nombres de columnas originales para el Pipeline
            input_df = pd.DataFrame([{
                'RevolvingUtilizationOfUnsecuredLines': uso, 
                'age': edad,
                'NumberOfTime30-59DaysPastDueNotWorse': r30, 
                'DebtRatio': deu,
                'MonthlyIncome': ing, 
                'NumberOfOpenCreditLinesAndLoans': 5,
                'NumberOfTimes90DaysLate': r90, 
                'NumberRealEstateLoansOrLines': 1,
                'NumberOfTime60-89DaysPastDueNotWorse': r60, 
                'NumberOfDependents': dep
            }])
            
            # Procesamiento en cadena: Limpieza -> Escalado -> Selección de Features -> Modelo
            df_cleaned = limpiador.transform(input_df)
            df_scaled = pd.DataFrame(scaler.transform(df_cleaned), columns=df_cleaned.columns)
            df_final = df_scaled[model_features]
            
            # Inferencia
            prob = modelo.predict_proba(df_final)[0, 1]
            dec, mot, ico = clasificar_cliente(prob, uso, deu)
            
            st.markdown("---")
            st.markdown(f"## {ico} Resultado: {dec}")
            st.metric("Probabilidad de Impago Estimada", f"{prob:.2%}")
            if mot != "N/A": 
                st.warning(f"**Alerta de Riesgo:** {mot}")
            else:
                st.success("Operación recomendada por el sistema")