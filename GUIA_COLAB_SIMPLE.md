# 🚀 Guía Simple para Ejecutar en Google Colab

Esta guía te ayudará a ejecutar el pipeline completo de predicción de precios BNB en Google Colab sin complicaciones.

## ✅ Problemas Resueltos

1. **TypeError corregido** en `Codigo_GrupoBNB.py`
2. **Guía simplificada** con 3 opciones fáciles

---

## 📋 Métodos de Ejecución

### ⭐ MÉTODO 1: Con Scripts (RECOMENDADO - Más Simple)

Este es el método más directo y menos propenso a errores.

#### Paso 1: Preparar Archivos
Descarga estos 3 archivos del repositorio:
- `Codigo_GrupoBNB.py` (corregido)
- `enhanced_additions.py`
- `run_complete_pipeline.py`

#### Paso 2: Abrir Google Colab
1. Ve a https://colab.research.google.com/
2. Crea un nuevo notebook

#### Paso 3: Subir Archivos
```python
# CELDA 1: Subir archivos
from google.colab import files
uploaded = files.upload()
# Selecciona los 3 archivos .py cuando se abra el diálogo
```

#### Paso 4: Instalar Dependencias
```python
# CELDA 2: Instalar dependencias (2 minutos)
!pip install yfinance pandas numpy matplotlib seaborn scikit-learn tensorflow==2.12.0 joblib
```

#### Paso 5: Ejecutar Pipeline
```python
# CELDA 3: Ejecutar pipeline completo (30-40 minutos)
!python run_complete_pipeline.py
```

#### Paso 6: Descargar Resultados
```python
# CELDA 4: Descargar todos los resultados
from google.colab import files
import os

# Modelos
print("Descargando modelos...")
for file in os.listdir('models/'):
    files.download(f'models/{file}')

# Visualizaciones
print("Descargando visualizaciones...")
for file in os.listdir('outputs/'):
    if file.endswith('.png'):
        files.download(f'outputs/{file}')

# Métricas
print("Descargando métricas...")
files.download('metrics_optimized.csv')
files.download('projections_15_days.csv')

print("✅ ¡Todos los archivos descargados!")
```

---

### ⚡ MÉTODO 2: Solo Script Base (Más Rápido)

Si solo quieres ver el pipeline base sin optimización completa:

#### Paso 1: Subir Solo Un Archivo
```python
# CELDA 1: Subir archivo
from google.colab import files
uploaded = files.upload()
# Selecciona solo: Codigo_GrupoBNB.py
```

#### Paso 2: Instalar e Ejecutar
```python
# CELDA 2: Instalar dependencias
!pip install yfinance pandas numpy matplotlib seaborn scikit-learn tensorflow joblib

# CELDA 3: Ejecutar (10-15 minutos)
!python Codigo_GrupoBNB.py
```

#### Paso 3: Ver Resultados
```python
# CELDA 4: Listar outputs
!ls -lh outputs/
!ls -lh models/

# Descargar si quieres
from google.colab import files
files.download('metrics.csv')
```

---

### 📝 MÉTODO 3: Code Inline (Paso a Paso)

Si prefieres tener control total sobre cada paso:

#### Celda 1: Setup
```python
# Instalar dependencias
!pip install -q yfinance pandas numpy matplotlib seaborn scikit-learn tensorflow==2.12.0 joblib

# Imports
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import joblib
import random
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2

# Seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Crear directorios
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)
os.makedirs('scalers', exist_ok=True)

print('✅ Setup completo')
```

#### Celda 2: Descargar Datos
```python
# Parámetros
SYMBOL = 'BNB-USD'
START_DATE = '2022-01-13'
END_DATE = '2024-11-15'

print(f"Descargando {SYMBOL}...")

try:
    data = yf.download(SYMBOL, start=START_DATE, end=END_DATE, progress=False)
    data.reset_index(inplace=True)
    print(f"✅ Datos reales: {len(data)} registros")
except:
    print("⚠️ Generando datos sintéticos...")
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    np.random.seed(42)
    
    prices = [220]
    for i in range(1, len(dates)):
        change = np.random.normal(0.001, 0.025)
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 50))
    
    data = pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * (1 + abs(np.random.normal(0, 0.015))) for p in prices],
        'Low': [p * (1 - abs(np.random.normal(0, 0.015))) for p in prices],
        'Close': prices,
        'Adj Close': prices,
        'Volume': [np.random.uniform(1e8, 5e8) for _ in prices]
    })
    print(f"✅ Datos sintéticos: {len(data)} registros")

print(f"Rango: ${data['Close'].min():.2f} - ${data['Close'].max():.2f}")
display(data.head())
```

#### Celda 3: Feature Engineering
```python
# Crear features
df = data[['Date', 'Close', 'High', 'Volume']].copy()
df['Daily_Return'] = df['Close'].pct_change()
df['Volatility'] = df['Daily_Return'].abs()
df = df.dropna().reset_index(drop=True)

print(f"✅ Features creadas: {len(df)} registros")
display(df.head())
```

#### Celda 4: EDA
```python
# Estadísticas
print("="*60)
print("ESTADÍSTICAS DESCRIPTIVAS - 5 VARIABLES")
print("="*60)

for col in ['Close', 'High', 'Volume', 'Daily_Return', 'Volatility']:
    print(f"\n{col}:")
    print(f"  Media: {df[col].mean():.6f}")
    print(f"  Std: {df[col].std():.6f}")
    print(f"  Min: {df[col].min():.6f}")
    print(f"  Max: {df[col].max():.6f}")
```

#### Celda 5: Visualizaciones
```python
# Series temporales
fig, axes = plt.subplots(5, 1, figsize=(14, 12))
variables = ['Close', 'High', 'Volume', 'Daily_Return', 'Volatility']
colors = ['blue', 'green', 'red', 'purple', 'orange']

for ax, var, color in zip(axes, variables, colors):
    ax.plot(df.index, df[var], color=color, linewidth=1.5)
    ax.set_title(f'{var} a lo largo del tiempo')
    ax.set_xlabel('Índice')
    ax.set_ylabel(var)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/time_series_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Correlación
plt.figure(figsize=(10, 8))
corr = df[['Close', 'High', 'Volume', 'Daily_Return', 'Volatility']].corr()
sns.heatmap(corr, annot=True, fmt='.3f', cmap='coolwarm', center=0)
plt.title('Matriz de Correlación - 5 Variables')
plt.tight_layout()
plt.savefig('outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()
```

#### Celdas 6+: División de Datos, Modelos, etc.

**Nota**: Para el pipeline completo con optimización, es mejor usar el **MÉTODO 1** que ejecuta `run_complete_pipeline.py`, ya que implementa todos los ~60 experimentos de tuning automáticamente.

---

## 📊 Outputs Esperados

### Método 1 (Completo - 30-40 min):
- ✅ 4 modelos optimizados (baseline, LSTM, CNN, LSTM-multi)
- ✅ 3 visualizaciones de tuning (10 paneles c/u)
- ✅ 1 visualización de proyecciones (6 paneles)
- ✅ 2 archivos CSV (métricas + proyecciones)

### Método 2 (Base - 10-15 min):
- ✅ 4 modelos básicos
- ✅ Métricas básicas
- ✅ Visualizaciones de EDA

### Método 3 (Manual - Variable):
- ✅ Control paso a paso
- ✅ Outputs según lo que implementes

---

## ⚠️ Solución de Problemas

### Error: "TypeError: unsupported format string"
**Solución**: ✅ Ya está corregido en la versión actual de `Codigo_GrupoBNB.py`

### Error: "No module named 'tensorflow'"
```python
!pip install tensorflow==2.12.0
```

### Error: "yfinance download failed"
- Normal en algunos entornos
- El código automáticamente usa datos sintéticos
- Los datos sintéticos funcionan perfectamente para demostrar el pipeline

### Session Timeout en Colab
Si el script se demora mucho (30-40 min), mantén la pestaña activa o:
```javascript
// Ejecutar esto en la consola del navegador (F12)
function KeepAlive() { 
  console.log("Keeping alive..."); 
  document.querySelector("colab-connect-button").click(); 
}
setInterval(KeepAlive, 60000);
```

### Poco espacio en Colab
Los modelos ocupan ~2MB, las visualizaciones ~4MB. Si hay problemas:
```python
# Eliminar archivos intermedios
!rm -rf __pycache__
!rm -rf .ipynb_checkpoints
```

---

## ✅ Checklist de Ejecución

**Método 1 (Recomendado)**:
- [ ] Descargar 3 archivos .py del repositorio
- [ ] Abrir Google Colab
- [ ] Subir los 3 archivos
- [ ] Instalar dependencias (Celda 2)
- [ ] Ejecutar `run_complete_pipeline.py` (Celda 3)
- [ ] Esperar 30-40 minutos
- [ ] Descargar resultados (Celda 4)
- [ ] ✅ LISTO

**Método 2 (Rápido)**:
- [ ] Descargar `Codigo_GrupoBNB.py`
- [ ] Subir a Colab
- [ ] Instalar + ejecutar
- [ ] Ver outputs en 10-15 min
- [ ] ✅ LISTO

---

## 🎯 Recomendación Final

**Para obtener TODOS los resultados** (optimización + proyecciones):
→ Usa **MÉTODO 1**

**Para ver el pipeline básico rápidamente**:
→ Usa **MÉTODO 2**

**Para aprender paso a paso**:
→ Usa **MÉTODO 3**

---

## 📞 ¿Necesitas Ayuda?

Si encuentras algún problema:
1. Verifica que hayas instalado las dependencias
2. Revisa que los archivos se hayan subido correctamente
3. Verifica la consola por mensajes de error
4. Asegúrate de tener suficiente tiempo de ejecución en Colab

---

**Última actualización**: 2024-11-20  
**Status**: ✅ Código funcional y probado  
**Errores conocidos**: ✅ Corregidos

¡Buena suerte con tu proyecto! 🚀
