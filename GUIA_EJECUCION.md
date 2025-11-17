# Guía de Ejecución - Pipeline BNB

## Opciones de Ejecución

### Opción 1: Ejecución Local

```bash
# 1. Clonar repositorio
git clone https://github.com/marleyyyocode/marleyyyocode.git
cd marleyyyocode

# 2. Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Ejecutar pipeline
python Codigo_GrupoBNB.py
```

**Tiempo estimado**: 10-20 minutos

### Opción 2: Google Colab (Recomendado)

1. Abre [Google Colab](https://colab.research.google.com/)
2. Sube el archivo `Notebook_GrupoBNB.ipynb`
3. Ejecuta las celdas en orden

**Ventajas**:
- No requiere instalación local
- Acceso a GPU gratuito
- Mejor conectividad con Yahoo Finance

### Opción 3: Jupyter Notebook Local

```bash
# Instalar Jupyter
pip install jupyter

# Abrir notebook
jupyter notebook Notebook_GrupoBNB.ipynb
```

## Estructura de Salida

Después de ejecutar, se generarán:

```
marleyyyocode/
├── models/                      # 4 modelos entrenados
│   ├── baseline_model.pkl
│   ├── lstm_univariado.h5
│   ├── cnn_univariado.h5
│   └── lstm_multivariado.h5
│
├── scalers/                     # 3 scalers
│   ├── scaler_train.pkl
│   ├── scaler_val.pkl
│   └── scaler_test.pkl
│
├── outputs/                     # 11 visualizaciones
│   ├── time_series_plots.png
│   ├── correlation_heatmap.png
│   ├── close_volatility_plot.png
│   ├── loss_curve_*.png (3 archivos)
│   ├── predictions_*.png (4 archivos)
│   └── comparison_all_models.png
│
└── metrics.csv                  # Tabla comparativa
```

## Interpretación de Resultados

### Métricas (metrics.csv)

```csv
Model,MSE,MAE,R2
Baseline (Linear Regression),631.16,18.54,0.967
LSTM Univariado,2062.77,34.57,0.891
CNN Univariado,1228.17,26.73,0.935
LSTM Multivariado,4493.89,53.39,0.763
```

**Análisis**:
- **MSE más bajo = mejor**: Baseline tiene el mejor MSE
- **R² más alto = mejor**: Baseline explica 96.7% de la varianza
- **MAE**: Error promedio en USD

### Visualizaciones Clave

1. **comparison_all_models.png**: Compara todos los modelos vs valores reales
2. **time_series_plots.png**: Series temporales originales
3. **loss_curve_*.png**: Evolución del entrenamiento
4. **predictions_*.png**: Predicciones individuales por modelo

## Uso de Modelos Guardados

```python
from tensorflow.keras.models import load_model
import joblib
import numpy as np

# Cargar modelo y scaler
model = load_model('models/lstm_univariado.h5')
scaler = joblib.load('scalers/scaler_test.pkl')

# Preparar datos (últimos 30 días escalados)
# X_new = ... (shape: (1, 30, 1))

# Predecir
predictions_scaled = model.predict(X_new)

# Revertir escalado
# predictions_original = inverse_transform(predictions_scaled, scaler)
```

## Troubleshooting

### Problema: yfinance no descarga datos

**Solución**: El script automáticamente genera datos sintéticos para demostración. Para datos reales:

```python
# En Codigo_GrupoBNB.py, línea ~60
# Intenta actualizar yfinance
pip install --upgrade yfinance

# O ejecuta en Google Colab que generalmente tiene mejor acceso
```

### Problema: Error de memoria

**Solución**: Reduce parámetros en `Codigo_GrupoBNB.py`:

```python
EPOCHS = 50  # Reducir de 100
# En model.fit, reducir batch_size a 16
```

### Problema: TensorFlow no se instala

**Solución**:

```bash
# Instalar versión CPU
pip install tensorflow-cpu>=2.13.0
```

## Personalización

### Cambiar símbolo de criptomoneda

```python
# En Codigo_GrupoBNB.py
SYMBOL = 'ETH-USD'  # Ethereum
SYMBOL = 'BTC-USD'  # Bitcoin
```

### Ajustar horizonte de predicción

```python
# En Codigo_GrupoBNB.py
HORIZON = 10  # Predecir 10 días en lugar de 5
```

### Modificar hiperparámetros

```python
LEARNING_RATE = 0.01  # Aumentar learning rate
EPOCHS = 150          # Más épocas
TIMESTEPS = 60        # Ventana más grande
```

## Recursos Adicionales

- **Documento_GrupoBNB.md**: Análisis completo y conclusiones
- **README_BNB.md**: Documentación del proyecto
- **Codigo_GrupoBNB.py**: Código fuente documentado

## Contacto y Soporte

Para preguntas o problemas:
1. Revisa esta guía primero
2. Consulta el README_BNB.md
3. Abre un issue en GitHub

---

**¡Éxito en tus predicciones! 🚀**
