# 🏦 Scoring Hipotecario: De Datos a Decisiones

Este proyecto resuelve un problema crítico en banca: **¿A quién aprobamos una hipoteca?** He construido un sistema de IA que analiza 150.000 perfiles financieros para predecir el riesgo de impago con precisión quirúrgica.

# 📈 Resultados Reales

Tras entrenar y testear el modelo con más de 100.000 expedientes nuevos, estos son los números:

* **Capacidad de Predicción:** **Gini de 0.71** (Excelente nivel de discriminación).
* **Eficiencia Operativa:** El **88% de las solicitudes se aprueban automáticamente**, liberando al equipo de riesgos para los casos complejos.
* **Detección de Riesgo:** El modelo identifica **5.4 veces más mora** en los segmentos críticos que un análisis al azar.

# 🛠️ Ingeniería de Datos (Pipeline)

Para garantizar la integridad del dato antes del modelado, implementé:

* **Winsorizing:** Tratamiento de outliers en ingresos y utilización para evitar distorsiones estadísticas sin eliminar registros.
* **Imputación por Medianas Agrupadas:** Resolución de nulos en `IngresoMensual` basada en perfiles socioeconómicos para mantener la coherencia del dataset.
* **Filtros de Coherencia:** Eliminación de inconsistencias (registros duplicados y edades < 18 años).
* **Feature Selection:** Consolidación de variables de retraso para reducir la redundancia detectada en la matriz de Spearman.

# 🧠 El "Stack" Técnico

* **Modelado:** Ensamble de **XGBoost + CatBoost + LightGBM** (lo más top en la industria).
* **Optimización:** Hiperparámetros ajustados con **Optuna**.
* **Explicabilidad:** Uso de **SHAP** para entender por qué el modelo dice "No" (Pista: el sobreendeudamiento es la clave).

# 🌐 Aplicación en Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://proyimpagohipotecas.streamlit.app/)

La aplicación está dividida en los siguientes módulos operativos:

* **📊 Dashboard de Cartera:** Análisis masivo de solicitudes para supervisores de riesgos, con métricas de aprobación y distribución de perfiles.
* **📁 Panel de Gestión de Cartera** Consulta individual de expedientes procesados por el modelo.
* **📝 Simulador de Riesgo:** Evaluación en tiempo real para nuevas solicitudes hipotecarias.

# 📁 ¿Qué hay en este Repo?

* `01_data`: El motor de datos (101.503 registros evaluados).
* `02_notebooks`: Donde ocurre la magia (EDA y Modelado).
* `04_models`: El cerebro del proyecto (Archivos `.joblib`).
* `05_app`: **Dashboard interactivo en Streamlit** para uso en tiempo real.

# 👩‍💻 Sobre mí

👋 ¡Hola! Soy Raquel, Data Analyst – Business & Financial Analytics – Data Science
Me apasiona pillar datos desordenados y convertirlos en decisiones que sirvan para algo. Con más de 15 años de trayectoria financiera, este proyecto es mi forma de demostrar cómo la **IA aplicada** puede asegurar la integridad del dato y el retorno de la inversión aplicada al negocio.

Hablamos en: [LinkedIn](www.linkedin.com/in/raquelvadillo) 

---

# ⚠️ ¡Un segundo! (Advertencia)

Este es un **proyecto educativo y técnico**. Aunque utiliza técnicas de vanguardia y datos reales, su propósito es demostrar capacidades en Ciencia de Datos y desarrollo de herramientas de soporte a la decisión, no debe ser usado como asesoría financiera real sin la supervisión de un experto certificado.

# 📊 Dataset

* **Fuente:** [Kaggle - Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
* **Observaciones:** Información histórica de **150.000 clientes** con variables de utilización de crédito, edad, historial de retrasos y situación socioeconómica.
* **Variable Objetivo:** `Impago2Años` (Original: *SeriousDlqin2yrs*). Indica si el cliente incurrirá en mora en los próximos 2 años.
