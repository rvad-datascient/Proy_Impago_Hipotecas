# **Scoring Crediticio Automatizado con Inteligencia Artificial**

Este proyecto aborda un reto clave en banca: **evaluar la viabilidad hipotecaria y automatizar la aprobación de créditos**.  
He desarrollado un sistema de Inteligencia Artificial que analiza 150.000 perfiles financieros para predecir la probabilidad de impago, transformando datos crudos en un motor de decisiones automatizado.

---

## **Resultados de Negocio e Impacto Operativo**

Validado sobre más de 100.000 expedientes nuevos:

- **Eficiencia Operativa:** Automatización del **87,56%** de solicitudes de bajo riesgo, reduciendo la carga manual del equipo de riesgos.  
- **Gestión de Revisión:** Identificación del **7,04%** de expedientes que requieren revisión manual o avales.  
- **Mitigación del Riesgo:** El modelo detecta **5,4 veces más mora** en los segmentos críticos que un análisis aleatorio.  
- **Calidad Predictiva:** Coeficiente **Gini = 0.71**, nivel propio de modelos productivos en banca.

---

## **Arquitectura del Sistema**

El proyecto funciona como un **Batch Inference Pipeline**:

1. Procesa nuevas solicitudes (`cs-test.csv`).  
2. Aplica limpieza, escalado y selección de variables mediante componentes serializados.  
3. Genera un dataset final (`scoring_output.csv`).  
4. Alimenta automáticamente una aplicación en Streamlit para análisis operativo.

Toda la lógica está documentada en:  
`02_notebooks/02_Feature_ML_Impago_Hipotecas.ipynb`

---

## **Aplicación en Streamlit**

La aplicación está diseñada para simular un entorno real de un departamento de riesgos:

* **Dashboard de Cartera:** Monitorización de KPIs de riesgo, distribución de decisiones y planificación del equipo analista.

* **Panel de Gestión:** Asignación equilibrada de expedientes por gestor, filtros operativos y buscador directo por ID.

* **Simulador de Riesgo:** Evaluación instantánea de nuevas solicitudes combinando modelo predictivo y reglas de negocio.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://proyimpagohipotecas.streamlit.app/)

---

## **Ingeniería de Datos y Modelado**

Para garantizar la integridad del dato antes del modelado, implementé:

* **Winsorizing:** Tratamiento de outliers en ingresos y utilización para evitar distorsiones estadísticas sin eliminar registros.
* **Imputación por Medianas Agrupadas:** Resolución de nulos en `IngresoMensual` basada en perfiles socioeconómicos para mantener la coherencia del dataset.
* **Filtros de Coherencia:** Eliminación de inconsistencias (registros duplicados y edades < 18 años).
* **Feature Selection:** Consolidación de variables de retraso para reducir la redundancia detectada en la matriz de Spearman.

### **Modelado Predictivo**
- Ensamble de **XGBoost, CatBoost y LightGBM**.  
- Optimización con **Optuna**.  
- Explicabilidad con **SHAP**:  
  - 71,7% de los rechazos se explican por uso excesivo de crédito.  
  - 27,5% por morosidad previa.

---

## **Estructura del Repositorio**

- `01_data`: Datos procesados y motores de datos.  
- `02_notebooks`: EDA, modelado y pipeline de inferencia.  
- `04_models`: Componentes serializados (.joblib).  
- `05_app`: Código de la aplicación en Streamlit.

---

## **Sobre mí**

Soy Raquel, Data Analyst y Data Science Junior.  
Enfoco mis proyectos en garantizar la integridad del dato y maximizar el retorno de la inversión mediante soluciones basadas en IA.

Hablamos en: [LinkedIn](www.linkedin.com/in/raquelvadillo) 

---

## **Dataset**

* **Fuente:** [Kaggle - Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
* **Observaciones:** Información histórica de **150.000 clientes** con variables de utilización de crédito, edad, historial de retrasos y situación socioeconómica.
* **Variable Objetivo:** `Impago2Años` (Original: *SeriousDlqin2yrs*). Indica si el cliente incurrirá en mora en los próximos 2 años.

---

## **¡Un segundo! (Advertencia)**

Este es un **proyecto educativo y técnico**. Aunque utiliza técnicas de vanguardia y datos reales, su propósito es demostrar capacidades en Ciencia de Datos y desarrollo de herramientas de soporte a la decisión, no debe ser usado como asesoría financiera real sin la supervisión de un experto certificado.
