import streamlit as st
import pandas as pd
import joblib
import os
import sys
import importlib
import numpy as np
import plotly.express as px

# ==============================================================================
# 1. CONFIGURACIÓN DE RUTAS ENFOQUE GITHUB (SEGURO)
# ==============================================================================

# Definimos la raíz del proyecto de forma dinámica para GitHub y Local
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))

if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# --- ESTILO Y COLORES CORPORATIVOS ---
PALETA = ['#FF6200', '#757575', '#D32F2F']

# --- CARGA DINÁMICA DE LIMPIEZA ---
try:
    limpieza_mod = importlib.import_module("03_src.limpieza")
    LimpiezaBasica = limpieza_mod.LimpiezaBasica
except Exception as e:
    st.error(f"Error crítico: No se pudo cargar el módulo de limpieza en 03_src.")
    st.stop()

# --- CARGA DE MODELOS CON RUTA RELATIVA PARA GITHUB ---
# Usamos 'experimental_allow_widget_deps' para evitar que se quede pensando si Git cambia los archivos
@st.cache_resource(experimental_allow_widget_deps=True)
def load_models(root_path):
    path_models = os.path.join(root_path, "04_models")
    return (
        joblib.load(os.path.join(path_models, 'limpiador_v1.joblib')),
        joblib.load(os.path.join(path_models, 'scaler_standard_v1.joblib')),
        joblib.load(os.path.join(path_models, 'credit_scorecard_v1_G71.joblib')),
        joblib.load(os.path.join(path_models, 'model_features_v1.joblib'))
    )

# --- CARGA DE DATOS CON RUTA RELATIVA PARA GITHUB ---
@st.cache_data
def load_processed_data(root_path):
    path_csv = os.path.join(root_path, "01_data", "processed", "scoring_output.csv")
    if os.path.exists(path_csv):
        df = pd.read_csv(path_csv)
        if "Unnamed: 0" in df.columns:
            if "id" not in df.columns:
                df = df.rename(columns={"Unnamed: 0": "id"})
            else:
                df = df.drop(columns=["Unnamed: 0"])
        return df
    return None

# Ejecución de carga pasando la raíz como argumento (así la caché no se buclea)
limpiador, scaler, modelo, model_features = load_models(root_dir)
df_p = load_processed_data(root_dir)

# --- LÓGICA DE NEGOCIO: ASIGNACIÓN DE GESTORES (EQUILIBRADA) ---
if df_p is not None and 'gestor' not in df_p.columns:

    import math

    exp_por_gestor = 60
    gestores_necesarios = max(1, math.ceil(len(df_p) / exp_por_gestor))
    lista_gestores = [f"Gestor {i}" for i in range(1, gestores_necesarios + 1)]

    # Mezclar expedientes para evitar sesgos
    df_p = df_p.sample(frac=1, random_state=42).reset_index(drop=True)

    # Crear columna vacía
    df_p['gestor'] = None

    # Reparto equilibrado por tipo de decisión
    for decision in df_p['decision'].unique():
        mask = df_p['decision'] == decision
        subset = df_p.loc[mask].copy().reset_index(drop=True)

        # Asignación equilibrada
        subset['gestor'] = [
            lista_gestores[i % gestores_necesarios]
            for i in range(len(subset))
        ]

        # Volver a colocar los valores en df_p respetando el orden original
        df_p.loc[mask, 'gestor'] = subset['gestor'].values


# ==============================================================================
# 2. LÓGICA DE NEGOCIO Y CONFIGURACIÓN UI
# ==============================================================================

def clasificar_cliente(score, uso_credito, debt_ratio):
    """Política de Riesgos: Criterios SHAP detectados en el Notebook 02"""
    if uso_credito > 0.9:
        motivo = "Exceso de uso en tarjetas"
    elif debt_ratio > 0.6:
        motivo = "Sobreendeudamiento"
    else:
        motivo = "Riesgo por perfil histórico"

    if score < 0.30: return "Pre-concedido (Campaña)", "Perfil de Riesgo Bajo", "🟢"
    elif score < 0.50: return "Requiere Aval / Garantía", "Reforzar solvencia", "🟡"
    else: return "Denegado", motivo, "🔴"

# --- CONFIGURACIÓN DE LA PÁGINA Y ENCABEZADO ---
st.set_page_config(
    page_title="Reporting Producción Bancarias",
    layout="wide",
    page_icon="🏦"
)

st.title("🏦 Reporting de Producción Bancarias")
st.markdown("---")
st.info("Panel de control de Scoring de Riesgos | Departamento de Análisis de Crédito")

# --- SIDEBAR ---
st.sidebar.markdown("## 🏦 Sistema de Scoring Hipotecas")
st.sidebar.markdown("---")
st.sidebar.title("Navegación")
menu = st.sidebar.radio(
    "Seleccione módulo:",
    ["📊 Dashboard de Cartera", "📁 Panel de Gestión de Cartera", "📝 Simulador de Riesgo"]

)
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Analítica de Riesgos - Uso Interno")

# ==============================================================================
# 3. MÓDULOS DE LA APLICACIÓN
# ==============================================================================

# --- MÓDULO 1: DASHBOARD TEAM LEADER ---
if menu == "📊 Dashboard de Cartera":
    st.title("📊 Dashboard Team Leader - Estado Global")

    if df_p is not None:
        try:
            # ============================
            # 1. MÉTRICAS GLOBALES
            # ============================
            total_solicitudes = len(df_p)
            orden_decisiones = ["Pre-concedido (Campaña)", "Requiere Aval / Garantía", "Denegado"]
            conteo = df_p["decision"].value_counts().reindex(orden_decisiones).fillna(0)
            pct = (conteo / total_solicitudes) * 100

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Solicitudes", f"{total_solicitudes:,}".replace(",", "."))
            m2.metric("Aprobado", f"{pct.iloc[0]:.1f}%", f"{int(conteo.iloc[0]):,} exp.")
            m3.metric("A Revisar", f"{pct.iloc[1]:.1f}%", f"{int(conteo.iloc[1]):,} exp.")
            m4.metric("Denegado", f"{pct.iloc[2]:.1f}%", f"{int(conteo.iloc[2]):,} exp.")

            st.markdown("---")

            # ============================
            # 2. PLANIFICACIÓN DE STAFF
            # ============================
            st.markdown("### 👥 Planificación y Reparto de Staff")

            exp_por_gestor = 60
            gestores_necesarios = round(total_solicitudes / exp_por_gestor)

            st.success(
                f"Gestores necesarios para procesar esta cartera: "
                f"**{max(1, gestores_necesarios):,} especialistas**."
            )

            st.markdown("---")

            # ============================
            # 3. GRÁFICO GLOBAL DE ESTADOS
            # ============================
            c1, c2 = st.columns([2, 1])
            with c1:
                df_plot = pd.DataFrame({"Estado": conteo.index, "Expedientes": conteo.values})

                COLORES_ESTADOS = {
                    "Pre-concedido (Campaña)": "#FF6200",   # Naranja
                    "Requiere Aval / Garantía": "#757575",  # Gris
                    "Denegado": "#D32F2F"                   # Rojo
                }

                fig = px.bar(
                    df_plot,
                    x="Estado",
                    y="Expedientes",
                    color="Estado",
                    color_discrete_map=COLORES_ESTADOS,
                    text_auto=',.0f',
                    title="PLAN DE ACCIÓN OPERATIVO"
                )
                fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Número de Clientes")
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                st.markdown("### 🎯 Drivers de Riesgo")
                st.write("**71.7%** de los rechazos se deben al **Uso de Crédito**.")
                st.write("**27.5%** responden a **Morosidad** previa.")
                st.divider()
                st.info("💡 **Estrategia sugerida:** Fomentar productos con aval para la 'Zona Gris'.")

            st.markdown("---")

            # ============================
            # 5. KPIs DE CARGA
            # ============================
            st.markdown("### 📊 Indicadores de Carga de Trabajo")

            df_dist = df_p.groupby("gestor").size()

            carga_media = int(df_dist.mean())
            carga_max = int(df_dist.max())
            carga_min = int(df_dist.min())

            k1, k2, k3 = st.columns(3)
            k1.metric("Carga media por gestor", f"{carga_media:,}".replace(",", ".") + " expedientes")
            k2.metric("Máxima carga detectada", f"{carga_max:,}".replace(",", ".") + " expedientes")
            k3.metric("Mínima carga detectada", f"{carga_min:,}".replace(",", ".") + " expedientes")

            # ============================
            # DESGLOSE DE DECISIONES (MEDIA POR GESTOR)
            # ============================
            df_tipo = (
                df_p.groupby(["gestor", "decision"])
                    .size()
                    .reset_index(name="expedientes")
            )

            # Media por tipo de decisión
            media_pre = int(df_tipo[df_tipo["decision"] == "Pre-concedido (Campaña)"]["expedientes"].mean())
            media_aval = int(df_tipo[df_tipo["decision"] == "Requiere Aval / Garantía"]["expedientes"].mean())
            media_den = int(df_tipo[df_tipo["decision"] == "Denegado"]["expedientes"].mean())

            st.markdown("#### 🔍 Desglose medio por tipo de decisión (por gestor)")
            st.write(f"• **Pre‑concedidos:** {media_pre:,}".replace(",", ".") + " por gestor")
            st.write(f"• **Requieren Aval:** {media_aval:,}".replace(",", ".") + " por gestor")
            st.write(f"• **Denegados:** {media_den:,}".replace(",", ".") + " por gestor")

            st.markdown("---")

            # ============================
            # 6. DISTRIBUCIÓN GLOBAL (BOXPLOT)
            # ============================
            st.markdown("### 📦 Distribución global de carga entre gestores")

            fig_box = px.box(
                df_dist,
                points=False,
                title="Distribución de expedientes por gestor",
                color_discrete_sequence=["#FF6200"]
            )
            fig_box.update_layout(yaxis_title="Expedientes por gestor", xaxis_title="")
            st.plotly_chart(fig_box, use_container_width=True)

            st.markdown("---")

        except Exception as e:
            st.error(f"Error al procesar las métricas: {e}")




# --- MÓDULO 2: PANEL DE GESTIÓN DE CARTERA ---
elif menu == "📁 Panel de Gestión de Cartera":
    st.header("📁 Panel de Gestión de Cartera")

    if df_p is not None:

        # ============================
        # 1. SELECCIÓN DE GESTOR
        # ============================
        col_g1, col_g2 = st.columns([1, 3])
        with col_g1:
            # Selector dinámico de gestores ordenados numéricamente
            gestor_actual = st.selectbox(
                "Identifícate como gestor:",
                sorted(df_p['gestor'].unique(), key=lambda x: int(x.split()[1]))
            )

        st.write(f"Viendo la cartera de: **{gestor_actual}**")

        # Cartera completa del gestor (sin filtros de flujo)
        df_gestor = df_p[df_p['gestor'] == gestor_actual].copy()

        # ============================
        # 2. FILTRO DE FLUJO DE TRABAJO (SE APLICA ANTES DEL GRÁFICO)
        # ============================
        st.markdown("### 🧮 Flujo de trabajo sobre tu cartera")

        filtro_estado = st.radio(
            "Selecciona flujo de trabajo:",
            ["Todos", "Solicitar Aval", "Gestionar Documentación (Pre-aprobado)", "Contactar para Rechazo"],
            horizontal=True
        )

        # Partimos de la cartera del gestor y filtramos SOLO para la vista de trabajo
        df_gestion = df_gestor.copy()

        if filtro_estado == "Solicitar Aval":
            df_gestion = df_gestion[df_gestion['decision'].str.contains('Aval', na=False)]
        elif filtro_estado == "Gestionar Documentación (Pre-aprobado)":
            df_gestion = df_gestion[df_gestion['decision'].str.contains('Pre-concedido', na=False)]
        elif filtro_estado == "Contactar para Rechazo":
            df_gestion = df_gestion[df_gestion['decision'].str.contains('Denegado', na=False)]

        # ============================
        # 3. KPIs DEL GESTOR (FILTRADOS)
        # ============================
        st.markdown("### 📊 Resumen Operativo del Gestor (Filtrado)")

        total = len(df_gestion)
        pre = (df_gestion['decision'].str.contains("Pre-concedido", na=False)).sum()
        aval = (df_gestion['decision'].str.contains("Aval", na=False)).sum()
        den = (df_gestion['decision'].str.contains("Denegado", na=False)).sum()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Expedientes", f"{total:,}".replace(",", "."))
        m2.metric("Pre-aprobados", f"{(pre/total*100 if total>0 else 0):.1f}%", f"{pre:,} exp.")
        m3.metric("Requieren Aval", f"{(aval/total*100 if total>0 else 0):.1f}%", f"{aval:,} exp.")
        m4.metric("Denegados", f"{(den/total*100 if total>0 else 0):.1f}%", f"{den:,} exp.")

        st.markdown("---")

        # ============================
        # 4. MINI DASHBOARD VISUAL (FILTRADO)
        # ============================

        COLORES_ESTADOS = {
            "Pre-concedido (Campaña)": "#FF6200",   # Naranja
            "Requiere Aval / Garantía": "#757575",  # Gris
            "Denegado": "#D32F2F"                   # Rojo
        }

        st.markdown("### 📈 Distribución de Estados (Filtrada)")

        if total > 0:
            fig = px.pie(
                df_gestion,
                names="decision",
                color="decision",
                color_discrete_map=COLORES_ESTADOS,
                hole=0.45,
                title="Distribución de Expedientes por Estado (Filtrado)"
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No hay expedientes para mostrar en este filtro.")

        st.markdown("---")

        # ============================
        # 5. PRIORIDAD OPERATIVA
        # ============================
        def prioridad(dec):
            if "Denegado" in dec:
                return "Alta"
            if "Aval" in dec:
                return "Media"
            return "Baja"

        if len(df_gestion) > 0:
            df_gestion["prioridad"] = df_gestion["decision"].apply(prioridad)

            mapa_prioridad = {"Alta": 1, "Media": 2, "Baja": 3}
            df_gestion["prioridad_num"] = df_gestion["prioridad"].map(mapa_prioridad)

            df_gestion = df_gestion.sort_values("prioridad_num")

            # ============================
            # 6. TABLA COLOREADA
            # ============================
            columnas_finales = ['id', 'IngresoMensual', 'score', 'decision', 'motivo_principal', 'prioridad']
            columnas_visibles = [c for c in columnas_finales if c in df_gestion.columns]

            st.markdown("### 📑 Expedientes Prioritarios en este flujo")

            def color_prioridad(row):
                if row["prioridad"] == "Alta":
                    return ['background-color: #FFCDD2'] * len(row)
                elif row["prioridad"] == "Media":
                    return ['background-color: #FFF9C4'] * len(row)
                else:
                    return ['background-color: #C8E6C9'] * len(row)

            st.dataframe(
                df_gestion[columnas_visibles].style.apply(color_prioridad, axis=1),
                use_container_width=True
            )
        else:
            st.info("No hay expedientes en este flujo de trabajo para este gestor.")

        st.markdown("---")

        # ============================
        # 7. BUSCADOR AVANZADO POR ID
        # ============================
        st.markdown("### 🔎 Búsqueda directa por ID")

        id_busqueda = st.text_input("Introduce un ID para abrir expediente:")

        if id_busqueda:
            expediente = df_p[df_p['id'].astype(str) == id_busqueda]
            if not expediente.empty:
                st.success(f"Expediente encontrado para ID {id_busqueda}")
                st.write(expediente)
            else:
                st.error("El ID no existe en la base de datos.")


# --- MÓDULO 3: SIMULADOR DE RIESGO (REDISEÑADO) ---
elif menu == "📝 Simulador de Riesgo":
    st.title("📝 Simulador de Riesgo – Evaluación Instantánea")
    st.markdown("Introduce los datos del cliente para obtener una evaluación clara y visual del riesgo.")

    with st.form("form_inferencia"):
        c1, c2 = st.columns(2)

        with c1:
            edad = st.number_input("Edad del cliente", 18, 90, 40)
            ingreso = st.number_input("Ingreso mensual (€)", 0, 50000, 3000)
            uso_credito = st.slider(
                "Uso de crédito (%)",
                0.0, 1.2, 0.3,
                help="Porcentaje de utilización de líneas de crédito. >0.9 implica riesgo muy alto."
            )

        with c2:
            ratio_deuda = st.slider(
                "Ratio de deuda (%)",
                0.0, 1.2, 0.35,
                help="Deuda total / ingresos. >0.6 implica sobreendeudamiento."
            )
            morosidad = st.number_input(
                "Número de retrasos +90 días",
                0, 10, 0,
                help="Historial de morosidad grave."
            )
            dependientes = st.number_input("Número de dependientes", 0, 10, 0)

        btn = st.form_submit_button("Calcular Riesgo")

    if btn:
        # 1. Preparar datos
        input_data = pd.DataFrame([{
            'UsoCrédito': uso_credito,
            'Edad': edad,
            '30-59DíasTarde': 0,
            '60-89DíasTarde': 0,
            '90DíasTarde': morosidad,
            'RatioDeuda': ratio_deuda,
            'IngresoMensual': ingreso,
            'LíneasCrédito': 5,
            'PréstamosCasa': 1,
            'Dependientes': dependientes
        }])

        # 2. Escalar y predecir
        X_scaled = pd.DataFrame(scaler.transform(input_data), columns=input_data.columns)
        X_final = X_scaled[model_features]
        prob = modelo.predict_proba(X_final)[0, 1]

        # 3. Clasificación con reglas de negocio
        if uso_credito > 0.9:
            decision = "Denegado"
            motivo = "Exceso de uso de crédito (>90%)"
            icono = "🔴"
        elif ratio_deuda > 0.6:
            decision = "Denegado"
            motivo = "Sobreendeudamiento (Ratio deuda > 60%)"
            icono = "🔴"
        elif prob > 0.5:
            decision = "Denegado"
            motivo = "Riesgo alto según modelo"
            icono = "🔴"
        elif prob > 0.3:
            decision = "Requiere Aval / Garantía"
            motivo = "Riesgo medio"
            icono = "🟡"
        else:
            decision = "Pre-concedido (Campaña)"
            motivo = "Riesgo bajo"
            icono = "🟢"

        # 4. Mostrar resultado
        st.markdown("---")
        st.subheader(f"{icono} Dictamen Final: **{decision}**")
        st.write(f"**Motivo principal:** {motivo}")

        colA, colB = st.columns(2)
        with colA:
            st.metric("Riesgo estimado (score)", f"{prob:.2%}")
        with colB:
            st.metric("Ingreso mensual", f"{ingreso:,.0f} €")

        # 5. Explicabilidad simplificada
        st.markdown("### 🔍 Factores que afectan al riesgo")

        if uso_credito > 0.9:
            st.error("• Uso de crédito extremadamente alto (>90%)")
        elif uso_credito > 0.6:
            st.warning("• Uso de crédito elevado (>60%)")

        if ratio_deuda > 0.6:
            st.error("• Ratio de deuda muy alto (>60%)")
        elif ratio_deuda > 0.4:
            st.warning("• Ratio de deuda moderado (>40%)")

        if morosidad > 0:
            st.error("• Historial de morosidad grave")

        if ingreso < 1200:
            st.warning("• Ingreso mensual bajo (<1200 €)")

        st.markdown("---")

        # 6. Recomendación operativa
        st.markdown("### 💡 Recomendación operativa")

        if decision == "Pre-concedido (Campaña)":
            st.success("Cliente apto. Proceder con documentación.")
        elif decision == "Requiere Aval / Garantía":
            st.warning("Solicitar avalista o garantías adicionales.")
        else:
            st.error("No recomendado según política de riesgos actual.")
