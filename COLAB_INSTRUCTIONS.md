# 🚀 Guía Rápida para Ejecutar en Google Colab

## Opción 1: Notebook Simplificado (RECOMENDADO)

He creado un notebook simplificado que puedes usar directamente en Google Colab.

### Pasos:

1. **Subir a Colab**:
   - Ve a https://colab.research.google.com/
   - File → Upload notebook
   - Sube `Notebook_GrupoBNB_Complete.ipynb`

2. **Ejecutar**:
   - Runtime → Run all
   - Espera 30-40 minutos

3. **Resultados**:
   - Los resultados se descargan automáticamente al final

## Opción 2: Ejecutar Código Base

Si prefieres trabajar con el código base, estos son los pasos:

### 1. Subir Archivos a Colab

Sube estos 3 archivos a tu sesión de Colab:
- `Codigo_GrupoBNB.py`
- `enhanced_additions.py`
- `run_complete_pipeline.py`

### 2. Instalar Dependencias

```python
!pip install yfinance pandas numpy matplotlib seaborn scikit-learn tensorflow joblib
```

### 3. Ejecutar Pipeline

```python
!python run_complete_pipeline.py
```

### 4. Descargar Resultados

```python
from google.colab import files

# Descargar modelos
files.download('models/baseline_model_optimized.pkl')
files.download('models/lstm_univariado_optimized.h5')
files.download('models/cnn_univariado_optimized.h5')
files.download('models/lstm_multivariado_optimized.h5')

# Descargar métricas
files.download('metrics_optimized.csv')
files.download('projections_15_days.csv')

# Descargar visualizaciones
import os
for file in os.listdir('outputs/'):
    if file.endswith('.png'):
        files.download(f'outputs/{file}')
```

## Opción 3: Código Inline en Colab

Para tener todo el código en celdas del notebook:

### Celda 1: Setup

```python
# Instalar dependencias
!pip install -q yfinance pandas numpy matplotlib seaborn scikit-learn tensorflow joblib

# Imports
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import joblib

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten

import random
import os
from datetime import datetime, timedelta

# Seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
import tensorflow as tf
tf.random.set_seed(SEED)

print('✅ Setup completo')
```

### Celda 2: Cargar Funciones

Copia el contenido de `Codigo_GrupoBNB.py` en una celda y ejecútala.

### Celda 3: Cargar Funciones Enhanced

Copia el contenido de `enhanced_additions.py` en una celda y ejecútala.

### Celda 4: Ejecutar Pipeline

```python
# Descargar datos
df = download_data()

# Feature engineering
df = feature_engineering(df)

# EDA
exploratory_data_analysis(df)

# Cross-validation
compare_cv_strategies(df)

# Splits
train_df, val_df, test_df = temporal_split(df)

# Escalado
train_scaled, val_scaled, test_scaled, scaler_train, scaler_val, scaler_test = scale_data(
    train_df, val_df, test_df
)

# Generar secuencias
X_train_univ, y_train = create_sequences_univariate(train_scaled['Scaled_Close'].values)
X_val_univ, y_val = create_sequences_univariate(val_scaled['Scaled_Close'].values)
X_test_univ, y_test = create_sequences_univariate(test_scaled['Scaled_Close'].values)

# ... continuar con el resto del pipeline
```

## 📊 Outputs Esperados

Después de la ejecución completa, obtendrás:

### Modelos (4 archivos)
- `baseline_model_optimized.pkl` (3-4 KB)
- `lstm_univariado_optimized.h5` (400 KB)
- `cnn_univariado_optimized.h5` (700 KB)
- `lstm_multivariado_optimized.h5` (400 KB)

### Visualizaciones (4+ archivos PNG)
- `tuning_lstm_univariado.png` - 10 paneles de análisis
- `tuning_cnn_univariado.png` - 10 paneles de análisis
- `tuning_lstm_multivariado.png` - 10 paneles de análisis
- `projections_all_models.png` - 6 paneles de proyecciones

### Métricas (2 archivos CSV)
- `metrics_optimized.csv` - MSE, MAE, R² por modelo
- `projections_15_days.csv` - 15 días proyectados

## ⏱️ Tiempo de Ejecución

| Fase | Duración |
|------|----------|
| Setup + descarga datos | 1-2 min |
| EDA + CV | 1-2 min |
| LSTM Univariado tuning | 10-12 min |
| CNN Univariado tuning | 10-12 min |
| LSTM Multivariado tuning | 10-12 min |
| Evaluación + proyecciones | 3-5 min |
| **TOTAL** | **30-40 min** |

## 💡 Tips para Colab

1. **Usa GPU**: Runtime → Change runtime type → Hardware accelerator: GPU
2. **Keep alive**: Usa extensiones como "Colab Keep Alive" para evitar disconnection
3. **Guarda progreso**: Descarga resultados intermedios periódicamente
4. **Reduce experimentos**: Si tienes poco tiempo, reduce el número de experimentos en las funciones de tuning

## 🆘 Troubleshooting

### Error: "No module named 'yfinance'"
```python
!pip install yfinance
```

### Error: "API rate limit exceeded"
- El código tiene fallback automático a datos sintéticos
- Espera unos minutos y vuelve a intentar

### Error: "Out of memory"
- Runtime → Manage sessions → Terminate session
- Restart con GPU habilitada
- Reduce batch size en experimentos

### Error: "Session timeout"
- Usa "Colab Keep Alive" extension
- Descarga resultados intermedios
- Ejecuta en bloques más pequeños

## 📚 Documentación Adicional

Para más detalles, consulta:
- `EJECUTAR_PIPELINE_COMPLETO.md` - Guía completa de ejecución
- `IMPLEMENTATION_COMPLETE.md` - Resumen técnico completo
- `README_BNB.md` - Documentación del proyecto
- `Documento_GrupoBNB.md` - Análisis académico

## ✅ Checklist de Ejecución

- [ ] Dependencias instaladas
- [ ] Archivos subidos (si usas Opción 2)
- [ ] Seeds configurados (SEED=42)
- [ ] Pipeline ejecutándose
- [ ] Experimentos de tuning completados
- [ ] Visualizaciones generadas
- [ ] Proyecciones calculadas
- [ ] Resultados descargados

---

**¿Problemas?** Consulta `EJECUTAR_PIPELINE_COMPLETO.md` para troubleshooting detallado.

**¿Preguntas?** Revisa los comentarios en el código o la documentación académica en `Documento_GrupoBNB.md`.
