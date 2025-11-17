# Predicción de Precios BNB (Binance Coin) 📈

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)](https://www.tensorflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green)]()

Pipeline completo y reproducible para predicción de precios de Binance Coin (BNB) usando Machine Learning y Deep Learning.

## 🎯 Descripción del Proyecto

Este proyecto implementa un sistema de predicción de precios de criptomonedas que:
- Descarga datos históricos de BNB desde Yahoo Finance
- Realiza análisis exploratorio completo (EDA)
- Implementa feature engineering (Daily Return, Volatility)
- Entrena múltiples modelos (Baseline, LSTM, CNN)
- Predice precios 5 días hacia el futuro
- Genera métricas comparativas y visualizaciones

## 📊 Modelos Implementados

1. **Baseline - Regresión Lineal**: Modelo simple para comparación
2. **LSTM Univariado**: Red recurrente procesando solo Close
3. **CNN Univariado**: Red convolucional 1D para patrones locales
4. **LSTM Multivariado**: LSTM con múltiples features (High, Volume, Volatility)

## 🚀 Inicio Rápido

### Prerrequisitos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)
- Conexión a internet (para descargar datos)

### Instalación

1. **Clonar el repositorio**
```bash
git clone https://github.com/marleyyyocode/marleyyyocode.git
cd marleyyyocode
```

2. **Crear entorno virtual (recomendado)**
```bash
# En Windows
python -m venv venv
venv\Scripts\activate

# En Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. **Instalar dependencias**
```bash
pip install -r requirements.txt
```

### Ejecución

**Ejecutar el pipeline completo:**
```bash
python Codigo_GrupoBNB.py
```

El script ejecutará automáticamente:
1. Descarga de datos de BNB-USD (2022-01-13 a 2024-11-15)
2. Análisis exploratorio y visualizaciones
3. Feature engineering
4. División temporal (Train/Val/Test)
5. Entrenamiento de 4 modelos
6. Evaluación y comparación
7. Generación de gráficos y métricas

**Tiempo estimado de ejecución:** 10-20 minutos (dependiendo del hardware)

### Alternativa: Google Colab

Para ejecutar en Google Colab sin instalación local:

```python
# En una celda de Colab
!git clone https://github.com/marleyyyocode/marleyyyocode.git
%cd marleyyyocode
!pip install -r requirements.txt
!python Codigo_GrupoBNB.py
```

## 📁 Estructura de Archivos

```
marleyyyocode/
│
├── Codigo_GrupoBNB.py          # Script principal del pipeline
├── Documento_GrupoBNB.md        # Análisis y resultados detallados
├── README.md                    # Este archivo
├── requirements.txt             # Dependencias del proyecto
├── metrics.csv                  # Tabla comparativa de métricas (generado)
│
├── models/                      # Modelos entrenados (generado)
│   ├── baseline_model.pkl
│   ├── lstm_univariado.h5
│   ├── cnn_univariado.h5
│   └── lstm_multivariado.h5
│
├── scalers/                     # Scalers guardados (generado)
│   ├── scaler_train.pkl
│   ├── scaler_val.pkl
│   └── scaler_test.pkl
│
└── outputs/                     # Visualizaciones (generado)
    ├── time_series_plots.png
    ├── correlation_heatmap.png
    ├── close_volatility_plot.png
    ├── loss_curve_lstm_univariado.png
    ├── loss_curve_cnn_univariado.png
    ├── loss_curve_lstm_multivariado.png
    ├── predictions_baseline.png
    ├── predictions_lstm_univariado.png
    ├── predictions_cnn_univariado.png
    ├── predictions_lstm_multivariado.png
    └── comparison_all_models.png
```

## 📈 Resultados

Después de ejecutar el pipeline, encontrarás:

### Métricas Comparativas (`metrics.csv`)
Tabla con MSE, MAE y R² para cada modelo en el conjunto de test.

### Visualizaciones (`outputs/`)
- **Series temporales**: Close, High, Volume
- **Correlaciones**: Mapa de calor entre variables
- **Volatilidad**: Evolución de la volatilidad del mercado
- **Curvas de pérdida**: Train vs Validation loss por modelo
- **Predicciones**: Comparación visual de predicciones vs valores reales
- **Comparación general**: Todos los modelos en una gráfica

### Modelos Entrenados (`models/`)
Modelos listos para cargar y hacer predicciones.

## 🔧 Configuración

### Parámetros Principales

Puedes modificar estos parámetros en `Codigo_GrupoBNB.py`:

```python
# Datos
SYMBOL = 'BNB-USD'
START_DATE = '2022-01-13'
END_DATE = '2024-11-15'

# Secuencias
TIMESTEPS = 30          # Ventana de entrada (días)
HORIZON = 5             # Predicción a futuro (días)

# Hiperparámetros
LEARNING_RATE = 0.001   # Tasa de aprendizaje
EPOCHS = 100            # Número de épocas
DROPOUT_RATE = 0.2      # Tasa de dropout
L2_REG = 0.001          # Regularización L2
```

### División Temporal

- **Train**: 2022-01-13 a 2023-11-30 (~687 días)
- **Validation**: 2023-12-01 a 2024-02-28 (~90 días)
- **Test**: 2024-03-01 a 2024-11-15 (~260 días)

## 🧪 Uso de Modelos Entrenados

### Cargar Modelo

```python
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Cargar modelo
model = load_model('models/lstm_univariado.h5')
scaler = joblib.load('scalers/scaler_test.pkl')

# Preparar datos (últimos 30 días escalados)
# X = ... (shape: (1, 30, 1) para univariado)

# Predecir
predictions_scaled = model.predict(X)

# Revertir escalado (usar función del script)
# predictions_original = inverse_transform_predictions(predictions_scaled, scaler)
```

## 📊 Métricas de Evaluación

- **MSE (Mean Squared Error)**: Penaliza errores grandes
- **MAE (Mean Absolute Error)**: Error promedio absoluto
- **R² (Coefficient of Determination)**: Proporción de varianza explicada

## 🤝 Contribución

Este es un proyecto académico del Grupo BNB. Las contribuciones son bienvenidas:

1. Fork del repositorio
2. Crear una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit de cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir un Pull Request

## 📚 Referencias

- [TensorFlow - LSTM Documentation](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- [Understanding LSTMs](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Augmented RNNs](https://distill.pub/2016/augmented-rnns/)
- [TensorFlow Time Series Tutorial](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## 📝 Documentación Adicional

Para análisis detallados, interpretaciones y conclusiones, consulta:
- **[Documento_GrupoBNB.md](Documento_GrupoBNB.md)**: Análisis completo de resultados

## ⚠️ Disclaimer

Este proyecto es **únicamente con fines educativos y de investigación**. Los modelos predictivos de precios de criptomonedas NO deben usarse como única base para decisiones de inversión. Los mercados de criptomonedas son altamente volátiles e impredecibles.

**No somos asesores financieros.** Cualquier decisión de inversión es responsabilidad exclusiva del usuario.

## 🛠️ Dependencias Principales

- `pandas==2.0.3`: Manipulación de datos
- `numpy==1.24.3`: Operaciones numéricas
- `matplotlib==3.7.2`: Visualizaciones
- `seaborn==0.12.2`: Visualizaciones estadísticas
- `scikit-learn==1.3.0`: Modelos ML y métricas
- `tensorflow==2.13.0`: Deep Learning
- `yfinance==0.2.28`: Descarga de datos financieros
- `joblib==1.3.2`: Serialización de modelos

Ver `requirements.txt` para la lista completa.

## 🐛 Solución de Problemas

### Error de instalación de TensorFlow

Si tienes problemas instalando TensorFlow:
```bash
# Instalar versión CPU
pip install tensorflow-cpu==2.13.0
```

### Error de descarga de datos

Si yfinance no puede descargar datos:
- Verifica tu conexión a internet
- Comprueba que el símbolo 'BNB-USD' esté disponible
- Intenta con un rango de fechas diferente

### Error de memoria

Si el script consume demasiada memoria:
- Reduce el número de épocas (EPOCHS)
- Reduce el tamaño de batch en model.fit (ej. batch_size=16)
- Usa menos datos (reduce el rango de fechas)

## 📧 Contacto

**Grupo BNB**
- Email: [tu-email@ejemplo.com]
- LinkedIn: [tu-perfil-linkedin]

---

## 📜 Licencia

Este proyecto está bajo la Licencia MIT - ver archivo LICENSE para detalles.

---

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub!**

---

*Proyecto desarrollado como parte del curso de Machine Learning y Data Science.*
*Universidad: [Tu Universidad]*
*Año: 2024*
