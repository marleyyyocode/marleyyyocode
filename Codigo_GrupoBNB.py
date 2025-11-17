"""
Codigo_GrupoBNB.py
==================
Pipeline completo para predicción de precios de Binance Coin (BNB) a 5 días.

Este script implementa:
- Descarga de datos históricos de BNB desde Yahoo Finance
- Análisis exploratorio de datos (EDA)
- Feature engineering (Daily_Return, Volatility)
- División temporal de datos (Train/Val/Test)
- Escalado de features
- Generación de secuencias temporales
- Modelos: Baseline (Regresión Lineal), LSTM Univariado, CNN Univariado, LSTM Multivariado
- Evaluación y visualización de resultados

Autor: Grupo BNB
Fecha: 2024-11-17
"""

import os
import warnings
warnings.filterwarnings('ignore')

# Fijar semillas para reproducibilidad
import random
import numpy as np
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Importar librerías
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import joblib

# Configuración de matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Configuración global
SYMBOL = 'BNB-USD'
START_DATE = '2022-01-13'
END_DATE = '2024-11-15'
TIMESTEPS = 30
HORIZON = 5
LEARNING_RATE = 0.001  # Documentado: probamos 0.001, 0.01, 0.1 y seleccionamos 0.001
EPOCHS = 100
DROPOUT_RATE = 0.2
L2_REG = 0.001
KERNEL_SIZE = 3
NUM_FILTERS = 32


def download_data(symbol=SYMBOL, start=START_DATE, end=END_DATE):
    """
    Descarga datos históricos de Yahoo Finance.
    
    Args:
        symbol (str): Símbolo del activo (ej. 'BNB-USD')
        start (str): Fecha de inicio (formato YYYY-MM-DD)
        end (str): Fecha de fin (formato YYYY-MM-DD)
    
    Returns:
        pd.DataFrame: DataFrame con datos históricos
    """
    print(f"Descargando datos de {symbol} desde {start} hasta {end}...")
    data = yf.download(symbol, start=start, end=end, progress=False)
    data.reset_index(inplace=True)
    print(f"Datos descargados: {len(data)} registros")
    return data


def create_filtered_dataset(data):
    """
    Crea dataset filtrado con Date, Close, High, Volume.
    
    Args:
        data (pd.DataFrame): DataFrame original con todos los datos
    
    Returns:
        pd.DataFrame: DataFrame filtrado
    """
    df = data[['Date', 'Close', 'High', 'Volume']].copy()
    print(f"Dataset filtrado creado con columnas: {df.columns.tolist()}")
    return df


def exploratory_data_analysis(df):
    """
    Realiza análisis exploratorio de datos.
    Calcula estadísticas descriptivas y genera visualizaciones.
    
    Args:
        df (pd.DataFrame): DataFrame con datos
    """
    print("\n" + "="*80)
    print("ANÁLISIS EXPLORATORIO DE DATOS")
    print("="*80)
    
    # Estadísticas descriptivas
    print("\nEstadísticas descriptivas para Close:")
    print(df['Close'].describe())
    print(f"\nMedia: {df['Close'].mean():.2f}")
    print(f"Mediana: {df['Close'].median():.2f}")
    print(f"Desviación estándar: {df['Close'].std():.2f}")
    print(f"Mínimo: {df['Close'].min():.2f}")
    print(f"Máximo: {df['Close'].max():.2f}")
    
    print("\nEstadísticas descriptivas para High:")
    print(df['High'].describe())
    
    print("\nEstadísticas descriptivas para Volume:")
    print(df['Volume'].describe())
    
    # Gráficas de series temporales
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    axes[0].plot(df['Date'], df['Close'], color='blue', linewidth=1.5)
    axes[0].set_title('Serie Temporal - Precio de Cierre (Close)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Fecha')
    axes[0].set_ylabel('Precio Close (USD)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df['Date'], df['High'], color='green', linewidth=1.5)
    axes[1].set_title('Serie Temporal - Precio Máximo (High)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Fecha')
    axes[1].set_ylabel('Precio High (USD)')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(df['Date'], df['Volume'], color='orange', linewidth=1.5)
    axes[2].set_title('Serie Temporal - Volumen', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Fecha')
    axes[2].set_ylabel('Volumen')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/time_series_plots.png', dpi=300, bbox_inches='tight')
    print("\nGráfica guardada: outputs/time_series_plots.png")
    plt.close()
    
    # Mapa de correlación
    corr_matrix = df[['Close', 'High', 'Volume']].corr()
    print("\nMatriz de correlación:")
    print(corr_matrix)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Mapa de Correlación: Close, High, Volume', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('outputs/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print("Gráfica guardada: outputs/correlation_heatmap.png")
    plt.close()


def feature_engineering(df):
    """
    Crea features adicionales: Daily_Return y Volatility.
    
    Args:
        df (pd.DataFrame): DataFrame con datos
    
    Returns:
        pd.DataFrame: DataFrame con features adicionales
    """
    print("\n" + "="*80)
    print("FEATURE ENGINEERING")
    print("="*80)
    
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Daily_Return'].abs()  # Volatilidad como valor absoluto del retorno
    
    # Eliminar primer valor NaN
    df = df.dropna().reset_index(drop=True)
    
    print(f"Features creados: Daily_Return, Volatility")
    print(f"Registros después de eliminar NaNs: {len(df)}")
    
    # Visualizar Close y Volatility
    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    
    axes[0].plot(df['Date'], df['Close'], color='blue', linewidth=1.5)
    axes[0].set_title('Precio de Cierre (Close)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Fecha')
    axes[0].set_ylabel('Precio (USD)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(df['Date'], df['Volatility'], color='red', linewidth=1.5)
    axes[1].set_title('Volatilidad (Daily Volatility)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Fecha')
    axes[1].set_ylabel('Volatilidad')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/close_volatility_plot.png', dpi=300, bbox_inches='tight')
    print("Gráfica guardada: outputs/close_volatility_plot.png")
    plt.close()
    
    return df


def temporal_split(df):
    """
    Divide los datos en train, validation y test según fechas específicas.
    
    Train: 2022-01-13 — 2023-11-30
    Validation: 2023-12-01 — 2024-02-28
    Test: 2024-03-01 — 2024-11-15
    
    Args:
        df (pd.DataFrame): DataFrame con todos los datos
    
    Returns:
        tuple: (train_df, val_df, test_df)
    """
    print("\n" + "="*80)
    print("DIVISIÓN TEMPORAL DE DATOS")
    print("="*80)
    
    train_df = df[(df['Date'] >= '2022-01-13') & (df['Date'] <= '2023-11-30')].copy()
    val_df = df[(df['Date'] >= '2023-12-01') & (df['Date'] <= '2024-02-28')].copy()
    test_df = df[(df['Date'] >= '2024-03-01') & (df['Date'] <= '2024-11-15')].copy()
    
    print(f"Train: {len(train_df)} registros (2022-01-13 a 2023-11-30)")
    print(f"Validation: {len(val_df)} registros (2023-12-01 a 2024-02-28)")
    print(f"Test: {len(test_df)} registros (2024-03-01 a 2024-11-15)")
    
    return train_df, val_df, test_df


def scale_data(train_df, val_df, test_df):
    """
    Escala los datos usando MinMaxScaler.
    
    NOTA IMPORTANTE: Según las instrucciones, se aplica escalado por separado a cada conjunto.
    Esto significa que cada conjunto (train, val, test) tiene su propio scaler ajustado.
    En producción, típicamente se ajustaría solo con train y se transformarían val/test,
    pero aquí seguimos las instrucciones específicas del enunciado.
    
    Args:
        train_df, val_df, test_df (pd.DataFrame): DataFrames de cada conjunto
    
    Returns:
        tuple: DataFrames escalados y scalers usados
    """
    print("\n" + "="*80)
    print("ESCALADO DE DATOS")
    print("="*80)
    print("NOTA: Escalando cada conjunto por separado según instrucciones.")
    print("      Cada conjunto tiene su propio scaler ajustado.")
    
    features_to_scale = ['Close', 'High', 'Volume', 'Volatility']
    
    # Scalers para cada conjunto
    scaler_train = MinMaxScaler(feature_range=(0, 1))
    scaler_val = MinMaxScaler(feature_range=(0, 1))
    scaler_test = MinMaxScaler(feature_range=(0, 1))
    
    # Escalar train
    train_scaled = train_df.copy()
    train_scaled[['Scaled_Close', 'Scaled_High', 'Scaled_Volume', 'Scaled_Volatility']] = \
        scaler_train.fit_transform(train_df[features_to_scale])
    
    # Escalar val
    val_scaled = val_df.copy()
    val_scaled[['Scaled_Close', 'Scaled_High', 'Scaled_Volume', 'Scaled_Volatility']] = \
        scaler_val.fit_transform(val_df[features_to_scale])
    
    # Escalar test
    test_scaled = test_df.copy()
    test_scaled[['Scaled_Close', 'Scaled_High', 'Scaled_Volume', 'Scaled_Volatility']] = \
        scaler_test.fit_transform(test_df[features_to_scale])
    
    print(f"Columnas escaladas creadas: Scaled_Close, Scaled_High, Scaled_Volume, Scaled_Volatility")
    
    # Guardar scalers
    joblib.dump(scaler_train, 'scalers/scaler_train.pkl')
    joblib.dump(scaler_val, 'scalers/scaler_val.pkl')
    joblib.dump(scaler_test, 'scalers/scaler_test.pkl')
    print("Scalers guardados en carpeta 'scalers/'")
    
    return train_scaled, val_scaled, test_scaled, scaler_train, scaler_val, scaler_test


def create_sequences_univariate(data, scaled_col='Scaled_Close', timesteps=TIMESTEPS, horizon=HORIZON):
    """
    Crea secuencias para modelos univariados (solo Close).
    
    Args:
        data (pd.DataFrame): DataFrame con datos escalados
        scaled_col (str): Columna escalada a usar
        timesteps (int): Ventana de entrada
        horizon (int): Número de pasos a predecir
    
    Returns:
        tuple: (X, y) donde X es (n_samples, timesteps, 1) y y es (n_samples, horizon)
    """
    X, y = [], []
    values = data[scaled_col].values
    
    for i in range(len(values) - timesteps - horizon + 1):
        X.append(values[i:i+timesteps])
        y.append(values[i+timesteps:i+timesteps+horizon])
    
    X = np.array(X).reshape(-1, timesteps, 1)
    y = np.array(y)
    
    return X, y


def create_sequences_multivariate(data, timesteps=TIMESTEPS, horizon=HORIZON):
    """
    Crea secuencias para modelos multivariados.
    
    Entrada (X): High, Volume, Volatility (3 features)
    Salida (y): Close (5 días futuros)
    
    Args:
        data (pd.DataFrame): DataFrame con datos escalados
        timesteps (int): Ventana de entrada
        horizon (int): Número de pasos a predecir
    
    Returns:
        tuple: (X, y) donde X es (n_samples, timesteps, 3) y y es (n_samples, horizon)
    """
    X, y = [], []
    
    features = data[['Scaled_High', 'Scaled_Volume', 'Scaled_Volatility']].values
    target = data['Scaled_Close'].values
    
    for i in range(len(data) - timesteps - horizon + 1):
        X.append(features[i:i+timesteps])
        y.append(target[i+timesteps:i+timesteps+horizon])
    
    X = np.array(X)
    y = np.array(y)
    
    return X, y


def inverse_transform_predictions(predictions, scaler, feature_idx=0):
    """
    Revierte el escalado de predicciones.
    
    Args:
        predictions (np.array): Predicciones escaladas
        scaler: Scaler usado
        feature_idx (int): Índice de la feature (0 para Close)
    
    Returns:
        np.array: Predicciones en escala original
    """
    # Crear array con la forma correcta para inverse_transform
    n_samples = predictions.shape[0]
    horizon = predictions.shape[1] if len(predictions.shape) > 1 else 1
    
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    
    # Crear array con ceros para las otras features
    dummy = np.zeros((n_samples, scaler.n_features_in_))
    
    predictions_original = []
    for i in range(n_samples):
        for h in range(horizon if len(predictions.shape) > 1 else 1):
            dummy_copy = dummy[0:1].copy()
            dummy_copy[0, feature_idx] = predictions[i, h] if len(predictions.shape) > 1 else predictions[i]
            original = scaler.inverse_transform(dummy_copy)[0, feature_idx]
            predictions_original.append(original)
    
    predictions_original = np.array(predictions_original).reshape(n_samples, -1)
    
    return predictions_original


def train_baseline_model(X_train, y_train, X_val, y_val):
    """
    Entrena modelo baseline de Regresión Lineal.
    
    Args:
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
    
    Returns:
        model: Modelo entrenado
    """
    print("\n" + "="*80)
    print("ENTRENAMIENTO: BASELINE - REGRESIÓN LINEAL")
    print("="*80)
    
    # Aplanar X para regresión lineal
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)
    
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X_train_flat, y_train)
    
    print(f"Modelo baseline entrenado con {X_train.shape[0]} muestras")
    
    # Evaluar en validación
    val_pred = model.predict(X_val_flat)
    val_mse = mean_squared_error(y_val, val_pred)
    print(f"MSE en validación: {val_mse:.6f}")
    
    # Guardar modelo
    joblib.dump(model, 'models/baseline_model.pkl')
    print("Modelo guardado: models/baseline_model.pkl")
    
    return model


def build_lstm_univariate(timesteps=TIMESTEPS, horizon=HORIZON, learning_rate=LEARNING_RATE):
    """
    Construye modelo LSTM univariado.
    
    Arquitectura: 4 capas totales
    - LSTM(64) con dropout
    - LSTM(32) con dropout
    - Dense(32) con regularización L2
    - Dense(horizon)
    
    Args:
        timesteps (int): Ventana de entrada
        horizon (int): Pasos a predecir
        learning_rate (float): Tasa de aprendizaje
    
    Returns:
        model: Modelo compilado
    """
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, 
             input_shape=(timesteps, 1), kernel_regularizer=l2(L2_REG)),
        Dropout(DROPOUT_RATE),
        LSTM(32, activation='relu', kernel_regularizer=l2(L2_REG)),
        Dropout(DROPOUT_RATE),
        Dense(32, activation='relu', kernel_regularizer=l2(L2_REG)),
        Dense(horizon)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def build_cnn_univariate(timesteps=TIMESTEPS, horizon=HORIZON, learning_rate=LEARNING_RATE):
    """
    Construye modelo CNN (Conv1D) univariado.
    
    Arquitectura: 4 capas totales
    - Conv1D(32, kernel_size=3) con dropout
    - Conv1D(64, kernel_size=3) con dropout
    - Flatten
    - Dense(32) con regularización L2
    - Dense(horizon)
    
    Args:
        timesteps (int): Ventana de entrada
        horizon (int): Pasos a predecir
        learning_rate (float): Tasa de aprendizaje
    
    Returns:
        model: Modelo compilado
    """
    model = Sequential([
        Conv1D(NUM_FILTERS, kernel_size=KERNEL_SIZE, activation='relu', 
               input_shape=(timesteps, 1), kernel_regularizer=l2(L2_REG)),
        Dropout(DROPOUT_RATE),
        Conv1D(64, kernel_size=KERNEL_SIZE, activation='relu', 
               kernel_regularizer=l2(L2_REG)),
        Dropout(DROPOUT_RATE),
        Flatten(),
        Dense(32, activation='relu', kernel_regularizer=l2(L2_REG)),
        Dense(horizon)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def build_lstm_multivariate(timesteps=TIMESTEPS, n_features=3, horizon=HORIZON, 
                           learning_rate=LEARNING_RATE):
    """
    Construye modelo LSTM multivariado.
    
    Arquitectura: 4 capas totales
    - LSTM(64) con dropout
    - LSTM(32) con dropout
    - Dense(32) con regularización L2
    - Dense(horizon)
    
    Args:
        timesteps (int): Ventana de entrada
        n_features (int): Número de features de entrada
        horizon (int): Pasos a predecir
        learning_rate (float): Tasa de aprendizaje
    
    Returns:
        model: Modelo compilado
    """
    model = Sequential([
        LSTM(64, activation='relu', return_sequences=True, 
             input_shape=(timesteps, n_features), kernel_regularizer=l2(L2_REG)),
        Dropout(DROPOUT_RATE),
        LSTM(32, activation='relu', kernel_regularizer=l2(L2_REG)),
        Dropout(DROPOUT_RATE),
        Dense(32, activation='relu', kernel_regularizer=l2(L2_REG)),
        Dense(horizon)
    ])
    
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    return model


def train_deep_learning_model(model, X_train, y_train, X_val, y_val, model_name, epochs=EPOCHS):
    """
    Entrena un modelo de deep learning.
    
    Args:
        model: Modelo a entrenar
        X_train, y_train: Datos de entrenamiento
        X_val, y_val: Datos de validación
        model_name (str): Nombre del modelo
        epochs (int): Número de épocas
    
    Returns:
        tuple: (model, history)
    """
    print(f"\n{'='*80}")
    print(f"ENTRENAMIENTO: {model_name.upper()}")
    print(f"{'='*80}")
    print(f"Épocas: {epochs}")
    print(f"Muestras de entrenamiento: {X_train.shape[0]}")
    print(f"Muestras de validación: {X_val.shape[0]}")
    print(f"Shape de entrada: {X_train.shape[1:]}")
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=0  # Sin output durante entrenamiento
    )
    
    # Resultados finales
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    print(f"\nResultados finales:")
    print(f"  Loss (train): {final_train_loss:.6f}")
    print(f"  Loss (val): {final_val_loss:.6f}")
    
    # Guardar modelo
    model_path = f'models/{model_name.replace(" ", "_").lower()}.h5'
    model.save(model_path)
    print(f"Modelo guardado: {model_path}")
    
    return model, history


def plot_training_history(history, model_name):
    """
    Grafica la evolución de pérdidas durante el entrenamiento.
    
    Args:
        history: Historial de entrenamiento
        model_name (str): Nombre del modelo
    """
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title(f'Evolución de Pérdida - {model_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Época')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f'outputs/loss_curve_{model_name.replace(" ", "_").lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {filename}")
    plt.close()


def evaluate_model(y_true, y_pred, model_name):
    """
    Evalúa un modelo calculando MSE, MAE y R².
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        model_name (str): Nombre del modelo
    
    Returns:
        dict: Diccionario con métricas
    """
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n{model_name} - Métricas en Test:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²:  {r2:.6f}")
    
    return {'Model': model_name, 'MSE': mse, 'MAE': mae, 'R2': r2}


def plot_predictions(y_true, y_pred, model_name, dates=None):
    """
    Grafica valores reales vs predicciones.
    
    Args:
        y_true: Valores reales
        y_pred: Predicciones
        model_name (str): Nombre del modelo
        dates: Fechas correspondientes (opcional)
    """
    # Para predicciones de múltiples horizontes, tomamos el primer día
    if len(y_true.shape) > 1:
        y_true_plot = y_true[:, 0]
        y_pred_plot = y_pred[:, 0]
    else:
        y_true_plot = y_true
        y_pred_plot = y_pred
    
    plt.figure(figsize=(14, 6))
    
    if dates is not None:
        plt.plot(dates[:len(y_true_plot)], y_true_plot, 
                label='Valores Reales', color='blue', linewidth=2, alpha=0.7)
        plt.plot(dates[:len(y_pred_plot)], y_pred_plot, 
                label='Predicciones', color='orange', linewidth=2, alpha=0.7)
    else:
        plt.plot(y_true_plot, label='Valores Reales', color='blue', linewidth=2, alpha=0.7)
        plt.plot(y_pred_plot, label='Predicciones', color='orange', linewidth=2, alpha=0.7)
    
    plt.title(f'Predicciones vs Valores Reales - {model_name}', 
             fontsize=14, fontweight='bold')
    plt.xlabel('Índice / Fecha')
    plt.ylabel('Precio Close (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    filename = f'outputs/predictions_{model_name.replace(" ", "_").lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Gráfica guardada: {filename}")
    plt.close()


def main():
    """
    Función principal que ejecuta todo el pipeline.
    """
    print("\n" + "="*80)
    print("PIPELINE DE PREDICCIÓN DE PRECIOS BNB")
    print("="*80)
    print(f"Símbolo: {SYMBOL}")
    print(f"Período: {START_DATE} a {END_DATE}")
    print(f"Timesteps: {TIMESTEPS}")
    print(f"Horizonte de predicción: {HORIZON} días")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Épocas: {EPOCHS}")
    
    # 1. Descargar datos
    data = download_data()
    
    # 2. Crear dataset filtrado
    df = create_filtered_dataset(data)
    
    # 3. Análisis exploratorio
    exploratory_data_analysis(df)
    
    # 4. Feature engineering
    df = feature_engineering(df)
    
    # 5. División temporal
    train_df, val_df, test_df = temporal_split(df)
    
    # 6. Escalado
    train_scaled, val_scaled, test_scaled, scaler_train, scaler_val, scaler_test = \
        scale_data(train_df, val_df, test_df)
    
    # 7. Crear secuencias para modelos univariados
    print("\n" + "="*80)
    print("GENERACIÓN DE SECUENCIAS")
    print("="*80)
    
    X_train_univ, y_train_univ = create_sequences_univariate(train_scaled)
    X_val_univ, y_val_univ = create_sequences_univariate(val_scaled)
    X_test_univ, y_test_univ = create_sequences_univariate(test_scaled)
    
    print(f"Secuencias univariadas creadas:")
    print(f"  Train: X={X_train_univ.shape}, y={y_train_univ.shape}")
    print(f"  Val:   X={X_val_univ.shape}, y={y_val_univ.shape}")
    print(f"  Test:  X={X_test_univ.shape}, y={y_test_univ.shape}")
    
    # 8. Crear secuencias para modelos multivariados
    X_train_mult, y_train_mult = create_sequences_multivariate(train_scaled)
    X_val_mult, y_val_mult = create_sequences_multivariate(val_scaled)
    X_test_mult, y_test_mult = create_sequences_multivariate(test_scaled)
    
    print(f"\nSecuencias multivariadas creadas:")
    print(f"  Train: X={X_train_mult.shape}, y={y_train_mult.shape}")
    print(f"  Val:   X={X_val_mult.shape}, y={y_val_mult.shape}")
    print(f"  Test:  X={X_test_mult.shape}, y={y_test_mult.shape}")
    
    # 9. Entrenar Baseline
    baseline_model = train_baseline_model(X_train_univ, y_train_univ, 
                                         X_val_univ, y_val_univ)
    
    # 10. Entrenar LSTM Univariado
    lstm_univ_model = build_lstm_univariate()
    lstm_univ_model, lstm_univ_history = train_deep_learning_model(
        lstm_univ_model, X_train_univ, y_train_univ, 
        X_val_univ, y_val_univ, 'LSTM Univariado'
    )
    plot_training_history(lstm_univ_history, 'LSTM Univariado')
    
    # 11. Entrenar CNN Univariado
    cnn_univ_model = build_cnn_univariate()
    cnn_univ_model, cnn_univ_history = train_deep_learning_model(
        cnn_univ_model, X_train_univ, y_train_univ, 
        X_val_univ, y_val_univ, 'CNN Univariado'
    )
    plot_training_history(cnn_univ_history, 'CNN Univariado')
    
    # 12. Entrenar LSTM Multivariado
    lstm_mult_model = build_lstm_multivariate()
    lstm_mult_model, lstm_mult_history = train_deep_learning_model(
        lstm_mult_model, X_train_mult, y_train_mult, 
        X_val_mult, y_val_mult, 'LSTM Multivariado'
    )
    plot_training_history(lstm_mult_history, 'LSTM Multivariado')
    
    # 13. Evaluación en Test
    print("\n" + "="*80)
    print("EVALUACIÓN EN CONJUNTO DE TEST")
    print("="*80)
    
    # Predicciones
    X_test_univ_flat = X_test_univ.reshape(X_test_univ.shape[0], -1)
    pred_baseline = baseline_model.predict(X_test_univ_flat)
    pred_lstm_univ = lstm_univ_model.predict(X_test_univ, verbose=0)
    pred_cnn_univ = cnn_univ_model.predict(X_test_univ, verbose=0)
    pred_lstm_mult = lstm_mult_model.predict(X_test_mult, verbose=0)
    
    # Revertir escalado
    y_test_original = inverse_transform_predictions(y_test_univ, scaler_test, feature_idx=0)
    pred_baseline_original = inverse_transform_predictions(pred_baseline, scaler_test, feature_idx=0)
    pred_lstm_univ_original = inverse_transform_predictions(pred_lstm_univ, scaler_test, feature_idx=0)
    pred_cnn_univ_original = inverse_transform_predictions(pred_cnn_univ, scaler_test, feature_idx=0)
    pred_lstm_mult_original = inverse_transform_predictions(pred_lstm_mult, scaler_test, feature_idx=0)
    
    # Calcular métricas
    metrics = []
    metrics.append(evaluate_model(y_test_original, pred_baseline_original, 'Baseline (Linear Regression)'))
    metrics.append(evaluate_model(y_test_original, pred_lstm_univ_original, 'LSTM Univariado'))
    metrics.append(evaluate_model(y_test_original, pred_cnn_univ_original, 'CNN Univariado'))
    metrics.append(evaluate_model(y_test_original, pred_lstm_mult_original, 'LSTM Multivariado'))
    
    # Guardar métricas
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('metrics.csv', index=False)
    print(f"\nTabla de métricas guardada: metrics.csv")
    print(metrics_df.to_string(index=False))
    
    # 14. Visualizaciones de predicciones
    print("\n" + "="*80)
    print("GENERANDO VISUALIZACIONES")
    print("="*80)
    
    plot_predictions(y_test_original, pred_baseline_original, 'Baseline')
    plot_predictions(y_test_original, pred_lstm_univ_original, 'LSTM Univariado')
    plot_predictions(y_test_original, pred_cnn_univ_original, 'CNN Univariado')
    plot_predictions(y_test_original, pred_lstm_mult_original, 'LSTM Multivariado')
    
    # 15. Gráfica comparativa de todos los modelos
    plt.figure(figsize=(16, 8))
    
    # Usar solo el primer horizonte para simplificar la visualización
    plt.plot(y_test_original[:, 0], label='Valores Reales', 
            color='blue', linewidth=2.5, alpha=0.8)
    plt.plot(pred_baseline_original[:, 0], label='Baseline', 
            color='gray', linewidth=1.5, alpha=0.7, linestyle='--')
    plt.plot(pred_lstm_univ_original[:, 0], label='LSTM Univariado', 
            color='green', linewidth=1.5, alpha=0.7)
    plt.plot(pred_cnn_univ_original[:, 0], label='CNN Univariado', 
            color='red', linewidth=1.5, alpha=0.7)
    plt.plot(pred_lstm_mult_original[:, 0], label='LSTM Multivariado', 
            color='purple', linewidth=1.5, alpha=0.7)
    
    plt.title('Comparación de Todos los Modelos - Test Set', 
             fontsize=16, fontweight='bold')
    plt.xlabel('Índice')
    plt.ylabel('Precio Close (USD)')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    plt.savefig('outputs/comparison_all_models.png', dpi=300, bbox_inches='tight')
    print("Gráfica guardada: outputs/comparison_all_models.png")
    plt.close()
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*80)
    print("\nArchivos generados:")
    print("  - models/ (modelos entrenados)")
    print("  - scalers/ (scalers guardados)")
    print("  - outputs/ (gráficas)")
    print("  - metrics.csv (tabla comparativa)")
    print("\n¡Proceso finalizado!")


if __name__ == "__main__":
    main()
