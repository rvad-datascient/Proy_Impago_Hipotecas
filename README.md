
# 🏠 Predicción de Impagos Hipotecarios con Machine Learning

Este proyecto utiliza técnicas avanzadas de Machine Learning para predecir la probabilidad de que un cliente incurra en impagos hipotecarios.

Aunque se trata de un caso simulado, los datos provienen del conjunto **"Give Me Some Credit"** publicado en Kaggle, ampliamente utilizado para problemas de scoring crediticio.  
El proyecto abarca desde el análisis exploratorio hasta el despliegue de una app funcional mediante Streamlit.

---
## 📌 Objetivo

Desarrollar un modelo predictivo que permita a entidades financieras **anticipar impagos** en solicitudes de créditos hipotecarios, ayudando así en la gestión de riesgo y toma de decisiones.

---
## 🧰 Tecnologías utilizadas

- Python
- Scikit-learn
- XGBoost, LightGBM, CatBoost
- Optuna
- SMOTE
- Streamlit
- SHAP

## 📁 Estructura del proyecto
```
Proy_Impago_Hipotecas/
├── 03_notebooks/                          # Notebooks de análisis y transformación
│   ├── Env_0_Datos_+_EDA_...ipynb         # Exploración y EDA del dataset
│   └── Env_1_Limpieza_Datos_...ipynb      # Feature engineering y limpieza
│
├── 05_ml_project/                         # App en producción y componentes
│   ├── app.py                             # Aplicación Streamlit
│   ├── limpieza_basica.py                 # Clase para limpieza avanzada
│   ├── preprocesamiento.py                # Funciones de transformación y escalado
│   ├── modelo_voting_fixed.pkl            # Modelo VotingClassifier entrenado
│   ├── scaler_robust.pkl                  # Escalador robusto para inputs
│   └── requirements.txt                   # Librerías necesarias para producción
│
├── .gitignore                             # Exclusión de archivos/carpetas innecesarias
└── README.md                              # Descripción del proyecto
```
## 🧪 Modelos probados

Se evaluaron varios modelos de boosting:

- ✅ **CatBoost** (mejor equilibrio entre recall y precisión)
- ✅ **XGBoost** (mayor recall sin umbral ajustado)
- ✅ **LightGBM** (buena velocidad y rendimiento)
- ✅ **VotingClassifier** (mejor resultado combinado)

### 🔎 Mejor combinación (Voting XGB + CatBoost):
- Precision: 0.36
- Recall: 0.51
- F1-score: 0.42
- ROC AUC: 0.86

---

## 🚀 Ver la App


```bash
streamlit run app.py
```

La app se abrirá automáticamente en tu navegador.

---

## 📊 Dataset

- **Fuente**: Kaggle - [Give Me Some Credit Dataset](https://www.kaggle.com/c/GiveMeSomeCredit)
- **Observaciones**: 150.000 clientes con información financiera
- **Variable objetivo**: Impago2Años Nombre original:`SeriousDlqin2yrs` → indica si el cliente tuvo impagos en los próximos 2 años

---

## 🧠 ¿Qué incluye este proyecto?

- Limpieza avanzada de datos con clase personalizada (`LimpiezaBasica`)
- Preprocesamiento: winsorización + escalado robusto + SMOTE
- Comparación de modelos Boosting y VotingClassifier
- Optimización de hiperparámetros con Optuna
- Ajuste del umbral de decisión
- Aplicación de predicción con Streamlit

---
## 💡 Sobre Mí

👋 ¡Hola! Soy Raquel, profesional en transición hacia la Ciencia de Datos, con experiencia previa en análisis financiero y mejora de procesos. Me apasiona transformar datos en decisiones útiles y aplicar modelos predictivos para resolver problemas reales.

📊 Este proyecto es parte de mi portfolio y representa un caso completo de predicción de impagos hipotecarios. Incluye análisis exploratorio, limpieza de datos, entrenamiento de modelos con técnicas avanzadas (XGBoost, CatBoost, VotingClassifier), y despliegue de una app interactiva usando Streamlit.

🚀 Actualmente sigo profundizando en temas como interpretabilidad con SHAP, balanceo de clases con SMOTE, y desarrollo de soluciones de datos listas para producción.
