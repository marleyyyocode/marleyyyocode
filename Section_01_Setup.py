"""
================================================================================
SECTION 1: SETUP & DEPENDENCIES
================================================================================
Descripción: Instalar paquetes, importar librerías y configurar semillas
Tiempo estimado: 2-3 minutos
Outputs esperados: ✅ Confirmación de instalación y versiones
================================================================================
"""

# ============================================================================
# INSTALACIÓN DE DEPENDENCIAS
# ============================================================================
print("=" * 80)
print("SECCIÓN 1/13: INSTALACIÓN Y CONFIGURACIÓN")
print("=" * 80)

print("\n[1/3] Instalando dependencias...")
import subprocess
import sys

packages = [
    'yfinance',
    'pandas',
    'numpy',
    'matplotlib',
    'seaborn',
    'scikit-learn',
    'tensorflow',
    'joblib'
]

for package in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', package])

print("✅ Todas las dependencias instaladas correctamente\n")

# ============================================================================
# IMPORTACIÓN DE LIBRERÍAS
# ============================================================================
print("[2/3] Importando librerías...")

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

print("✅ Librerías importadas correctamente\n")

# ============================================================================
# CONFIGURACIÓN DE SEMILLAS (REPRODUCIBILIDAD)
# ============================================================================
print("[3/3] Configurando semillas para reproducibilidad...")

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

print(f"✅ SEED establecida: {SEED}\n")

# ============================================================================
# VERIFICACIÓN DE VERSIONES
# ============================================================================
print("=" * 80)
print("VERSIONES DE LIBRERÍAS INSTALADAS")
print("=" * 80)
print(f"Python: {sys.version.split()[0]}")
print(f"Pandas: {pd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"TensorFlow: {tf.__version__}")
print("=" * 80)

print("\n✅ SECCIÓN 1 COMPLETADA")
print("=" * 80)
print("Siguiente: Section_02_Data_Download.py")
print("=" * 80)
