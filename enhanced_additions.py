"""
Enhanced Model-Specific Tuning and Projections Module
=====================================================
This module extends Codigo_GrupoBNB.py with:
1. Individual hyperparameter optimization for each model
2. Projection capabilities for all models
3. Comprehensive visualization of tuning results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# ============================================================================
# HYPERPARAMETER TUNING FUNCTIONS - MODEL SPECIFIC
# ============================================================================

def tune_lstm_univariado(X_train, y_train, X_val, y_val, scaler_train):
    """
    Optimización individual de hiperparámetros para LSTM Univariado.
    Prueba múltiples configuraciones arquitectónicas.
    """
    print("\n" + "="*80)
    print("OPTIMIZACIÓN LSTM UNIVARIADO - Búsqueda Individual")
    print("="*80)
    
    # Grid de búsqueda selectivo pero comprehensivo
    search_space = {
        'learning_rate': [0.001, 0.005, 0.01],
        'architecture': [
            ('2layers', [64, 32]),
            ('3layers', [128, 64, 32]),
            ('4layers', [256, 128, 64, 32])
        ],
        'activation': ['relu', 'tanh'],
        'dropout': [0.1, 0.2, 0.3],
        'batch_size': [16, 32],
        'optimizer': ['Adam', 'RMSprop'],
        'loss': ['mse', 'mae'],
        'l2_reg': [0.0, 0.001]
    }
    
    # Experimentos seleccionados (diversidad máxima con ~20 experimentos)
    experiments = [
        {'lr': 0.001, 'arch': ('2layers', [64, 32]), 'act': 'relu', 'drop': 0.2, 'bs': 32, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.005, 'arch': ('3layers', [128, 64, 32]), 'act': 'relu', 'drop': 0.2, 'bs': 16, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.0},
        {'lr': 0.01, 'arch': ('4layers', [256, 128, 64, 32]), 'act': 'relu', 'drop': 0.1, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.0},
        {'lr': 0.001, 'arch': ('3layers', [128, 64, 32]), 'act': 'tanh', 'drop': 0.3, 'bs': 16, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.005, 'arch': ('2layers', [64, 32]), 'act': 'relu', 'drop': 0.1, 'bs': 32, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.0},
        {'lr': 0.01, 'arch': ('3layers', [128, 64, 32]), 'act': 'tanh', 'drop': 0.2, 'bs': 16, 'opt': 'Adam', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.001, 'arch': ('4layers', [256, 128, 64, 32]), 'act': 'relu', 'drop': 0.2, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.0},
        {'lr': 0.005, 'arch': ('4layers', [256, 128, 64, 32]), 'act': 'tanh', 'drop': 0.1, 'bs': 32, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.01, 'arch': ('2layers', [64, 32]), 'act': 'tanh', 'drop': 0.3, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.001, 'arch': ('3layers', [128, 64, 32]), 'act': 'relu', 'drop': 0.1, 'bs': 32, 'opt': 'Adam', 'loss': 'mae', 'l2': 0.0},
        {'lr': 0.005, 'arch': ('2layers', [64, 32]), 'act': 'tanh', 'drop': 0.2, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.01, 'arch': ('4layers', [256, 128, 64, 32]), 'act': 'relu', 'drop': 0.3, 'bs': 32, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.0},
        {'lr': 0.001, 'arch': ('2layers', [64, 32]), 'act': 'relu', 'drop': 0.3, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.005, 'arch': ('4layers', [256, 128, 64, 32]), 'act': 'tanh', 'drop': 0.2, 'bs': 32, 'opt': 'Adam', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.01, 'arch': ('3layers', [128, 64, 32]), 'act': 'relu', 'drop': 0.1, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.0},
        {'lr': 0.001, 'arch': ('4layers', [256, 128, 64, 32]), 'act': 'tanh', 'drop': 0.1, 'bs': 32, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.005, 'arch': ('3layers', [128, 64, 32]), 'act': 'relu', 'drop': 0.3, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.0},
        {'lr': 0.01, 'arch': ('2layers', [64, 32]), 'act': 'tanh', 'drop': 0.2, 'bs': 32, 'opt': 'Adam', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.001, 'arch': ('3layers', [128, 64, 32]), 'act': 'tanh', 'drop': 0.2, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.005, 'arch': ('4layers', [256, 128, 64, 32]), 'act': 'relu', 'drop': 0.1, 'bs': 32, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.0},
    ]
    
    results = []
    best_val_loss = float('inf')
    best_config = None
    best_model = None
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Testing: LR={exp['lr']}, Arch={exp['arch'][0]}, "
              f"Act={exp['act']}, Dropout={exp['drop']}, BS={exp['bs']}, "
              f"Opt={exp['opt']}, Loss={exp['loss']}, L2={exp['l2']}")
        
        # Construir modelo
        model = Sequential(name=f"LSTM_Univ_Exp{i}")
        units_list = exp['arch'][1]
        
        # Primera capa LSTM
        model.add(LSTM(units_list[0], activation=exp['act'], return_sequences=(len(units_list) > 1),
                      kernel_regularizer=l2(exp['l2']), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(exp['drop']))
        
        # Capas LSTM adicionales
        for j, units in enumerate(units_list[1:]):
            return_seq = (j < len(units_list) - 2)
            model.add(LSTM(units, activation=exp['act'], return_sequences=return_seq,
                          kernel_regularizer=l2(exp['l2'])))
            model.add(Dropout(exp['drop']))
        
        # Capa de salida
        model.add(Dense(5, kernel_regularizer=l2(exp['l2'])))
        
        # Compilar
        optimizer = Adam(learning_rate=exp['lr']) if exp['opt'] == 'Adam' else RMSprop(learning_rate=exp['lr'])
        model.compile(optimizer=optimizer, loss=exp['loss'], metrics=['mae', 'mse'])
        
        # Entrenar
        history = model.fit(X_train, y_train,
                          validation_data=(X_val, y_val),
                          epochs=80,
                          batch_size=exp['bs'],
                          verbose=0)
        
        # Evaluar
        val_loss = history.history['val_loss'][-1]
        val_mae = history.history['val_mae'][-1]
        
        results.append({
            'experiment': i,
            'lr': exp['lr'],
            'architecture': exp['arch'][0],
            'activation': exp['act'],
            'dropout': exp['drop'],
            'batch_size': exp['bs'],
            'optimizer': exp['opt'],
            'loss_fn': exp['loss'],
            'l2_reg': exp['l2'],
            'val_loss': val_loss,
            'val_mae': val_mae
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = exp
            best_model = model
            print(f"  → NUEVO MEJOR! Val Loss: {val_loss:.4f}")
        else:
            print(f"  → Val Loss: {val_loss:.4f}")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("RESULTADOS DE OPTIMIZACIÓN LSTM UNIVARIADO")
    print("="*80)
    print(f"\nMejor configuración (Val Loss={best_val_loss:.4f}):")
    for key, value in best_config.items():
        if key != 'arch':
            print(f"  {key}: {value}")
        else:
            print(f"  architecture: {value[0]} - units: {value[1]}")
    
    return best_model, best_config, results_df


def tune_cnn_univariado(X_train, y_train, X_val, y_val, scaler_train):
    """
    Optimización individual de hiperparámetros para CNN Univariado.
    """
    print("\n" + "="*80)
    print("OPTIMIZACIÓN CNN UNIVARIADO - Búsqueda Individual")
    print("="*80)
    
    # Experimentos seleccionados para CNN (arquitecturas convolucionales)
    experiments = [
        {'lr': 0.0001, 'filters': [32, 64], 'kernel': 3, 'act': 'relu', 'drop': 0.2, 'bs': 32, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.0005, 'filters': [64, 128], 'kernel': 5, 'act': 'relu', 'drop': 0.2, 'bs': 16, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.0},
        {'lr': 0.001, 'filters': [128, 256], 'kernel': 3, 'act': 'elu', 'drop': 0.1, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.0},
        {'lr': 0.0001, 'filters': [64, 128], 'kernel': 5, 'act': 'elu', 'drop': 0.3, 'bs': 16, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.0005, 'filters': [32, 64], 'kernel': 3, 'act': 'relu', 'drop': 0.1, 'bs': 32, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.0},
        {'lr': 0.001, 'filters': [64, 128], 'kernel': 5, 'act': 'elu', 'drop': 0.2, 'bs': 16, 'opt': 'Adam', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.0001, 'filters': [128, 256], 'kernel': 3, 'act': 'relu', 'drop': 0.2, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.0},
        {'lr': 0.0005, 'filters': [128, 256], 'kernel': 5, 'act': 'elu', 'drop': 0.1, 'bs': 32, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.001, 'filters': [32, 64], 'kernel': 3, 'act': 'elu', 'drop': 0.3, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.0001, 'filters': [64, 128], 'kernel': 3, 'act': 'relu', 'drop': 0.1, 'bs': 32, 'opt': 'Adam', 'loss': 'mae', 'l2': 0.0},
        {'lr': 0.0005, 'filters': [32, 64], 'kernel': 5, 'act': 'elu', 'drop': 0.2, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.001, 'filters': [128, 256], 'kernel': 3, 'act': 'relu', 'drop': 0.3, 'bs': 32, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.0},
        {'lr': 0.0001, 'filters': [32, 64], 'kernel': 5, 'act': 'relu', 'drop': 0.3, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.0005, 'filters': [128, 256], 'kernel': 3, 'act': 'elu', 'drop': 0.2, 'bs': 32, 'opt': 'Adam', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.001, 'filters': [64, 128], 'kernel': 5, 'act': 'relu', 'drop': 0.1, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.0},
        {'lr': 0.0001, 'filters': [128, 256], 'kernel': 5, 'act': 'elu', 'drop': 0.1, 'bs': 32, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.0005, 'filters': [64, 128], 'kernel': 3, 'act': 'relu', 'drop': 0.3, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.0},
        {'lr': 0.001, 'filters': [32, 64], 'kernel': 5, 'act': 'elu', 'drop': 0.2, 'bs': 32, 'opt': 'Adam', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.0001, 'filters': [64, 128], 'kernel': 3, 'act': 'elu', 'drop': 0.2, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.0005, 'filters': [128, 256], 'kernel': 5, 'act': 'relu', 'drop': 0.1, 'bs': 32, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.0},
    ]
    
    results = []
    best_val_loss = float('inf')
    best_config = None
    best_model = None
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Testing: LR={exp['lr']}, Filters={exp['filters']}, "
              f"Kernel={exp['kernel']}, Act={exp['act']}, Dropout={exp['drop']}, BS={exp['bs']}")
        
        # Construir modelo CNN
        model = Sequential(name=f"CNN_Univ_Exp{i}")
        
        # Capas convolucionales
        for j, filters in enumerate(exp['filters']):
            if j == 0:
                model.add(Conv1D(filters=filters, kernel_size=exp['kernel'], 
                                activation=exp['act'], kernel_regularizer=l2(exp['l2']),
                                input_shape=(X_train.shape[1], X_train.shape[2])))
            else:
                model.add(Conv1D(filters=filters, kernel_size=exp['kernel'],
                                activation=exp['act'], kernel_regularizer=l2(exp['l2'])))
            model.add(Dropout(exp['drop']))
        
        model.add(Flatten())
        model.add(Dense(64, activation=exp['act'], kernel_regularizer=l2(exp['l2'])))
        model.add(Dropout(exp['drop']))
        model.add(Dense(5, kernel_regularizer=l2(exp['l2'])))
        
        # Compilar
        optimizer = Adam(learning_rate=exp['lr']) if exp['opt'] == 'Adam' else RMSprop(learning_rate=exp['lr'])
        model.compile(optimizer=optimizer, loss=exp['loss'], metrics=['mae', 'mse'])
        
        # Entrenar
        history = model.fit(X_train, y_train,
                          validation_data=(X_val, y_val),
                          epochs=80,
                          batch_size=exp['bs'],
                          verbose=0)
        
        # Evaluar
        val_loss = history.history['val_loss'][-1]
        val_mae = history.history['val_mae'][-1]
        
        results.append({
            'experiment': i,
            'lr': exp['lr'],
            'filters': str(exp['filters']),
            'kernel_size': exp['kernel'],
            'activation': exp['act'],
            'dropout': exp['drop'],
            'batch_size': exp['bs'],
            'optimizer': exp['opt'],
            'loss_fn': exp['loss'],
            'l2_reg': exp['l2'],
            'val_loss': val_loss,
            'val_mae': val_mae
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = exp
            best_model = model
            print(f"  → NUEVO MEJOR! Val Loss: {val_loss:.4f}")
        else:
            print(f"  → Val Loss: {val_loss:.4f}")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("RESULTADOS DE OPTIMIZACIÓN CNN UNIVARIADO")
    print("="*80)
    print(f"\nMejor configuración (Val Loss={best_val_loss:.4f}):")
    for key, value in best_config.items():
        print(f"  {key}: {value}")
    
    return best_model, best_config, results_df


def tune_lstm_multivariado(X_train, y_train, X_val, y_val, scaler_train):
    """
    Optimización individual de hiperparámetros para LSTM Multivariado.
    Similar a LSTM univariado pero adaptado para múltiples features.
    """
    print("\n" + "="*80)
    print("OPTIMIZACIÓN LSTM MULTIVARIADO - Búsqueda Individual")
    print("="*80)
    
    # Experimentos para LSTM multivariado (similar a univariado)
    experiments = [
        {'lr': 0.001, 'arch': ('2layers', [64, 32]), 'act': 'relu', 'drop': 0.2, 'bs': 32, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.005, 'arch': ('3layers', [128, 64, 32]), 'act': 'relu', 'drop': 0.2, 'bs': 16, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.0},
        {'lr': 0.01, 'arch': ('4layers', [256, 128, 64, 32]), 'act': 'relu', 'drop': 0.1, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.0},
        {'lr': 0.001, 'arch': ('3layers', [128, 64, 32]), 'act': 'tanh', 'drop': 0.3, 'bs': 16, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.005, 'arch': ('2layers', [64, 32]), 'act': 'relu', 'drop': 0.1, 'bs': 32, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.0},
        {'lr': 0.01, 'arch': ('3layers', [128, 64, 32]), 'act': 'tanh', 'drop': 0.2, 'bs': 16, 'opt': 'Adam', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.001, 'arch': ('4layers', [256, 128, 64, 32]), 'act': 'relu', 'drop': 0.2, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.0},
        {'lr': 0.005, 'arch': ('4layers', [256, 128, 64, 32]), 'act': 'tanh', 'drop': 0.1, 'bs': 32, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.01, 'arch': ('2layers', [64, 32]), 'act': 'tanh', 'drop': 0.3, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.001, 'arch': ('3layers', [128, 64, 32]), 'act': 'relu', 'drop': 0.1, 'bs': 32, 'opt': 'Adam', 'loss': 'mae', 'l2': 0.0},
        {'lr': 0.005, 'arch': ('2layers', [64, 32]), 'act': 'tanh', 'drop': 0.2, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.01, 'arch': ('4layers', [256, 128, 64, 32]), 'act': 'relu', 'drop': 0.3, 'bs': 32, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.0},
        {'lr': 0.001, 'arch': ('2layers', [64, 32]), 'act': 'relu', 'drop': 0.3, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.005, 'arch': ('4layers', [256, 128, 64, 32]), 'act': 'tanh', 'drop': 0.2, 'bs': 32, 'opt': 'Adam', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.01, 'arch': ('3layers', [128, 64, 32]), 'act': 'relu', 'drop': 0.1, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.0},
        {'lr': 0.001, 'arch': ('4layers', [256, 128, 64, 32]), 'act': 'tanh', 'drop': 0.1, 'bs': 32, 'opt': 'Adam', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.005, 'arch': ('3layers', [128, 64, 32]), 'act': 'relu', 'drop': 0.3, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mae', 'l2': 0.0},
        {'lr': 0.01, 'arch': ('2layers', [64, 32]), 'act': 'tanh', 'drop': 0.2, 'bs': 32, 'opt': 'Adam', 'loss': 'mae', 'l2': 0.001},
        {'lr': 0.001, 'arch': ('3layers', [128, 64, 32]), 'act': 'tanh', 'drop': 0.2, 'bs': 16, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.001},
        {'lr': 0.005, 'arch': ('4layers', [256, 128, 64, 32]), 'act': 'relu', 'drop': 0.1, 'bs': 32, 'opt': 'RMSprop', 'loss': 'mse', 'l2': 0.0},
    ]
    
    results = []
    best_val_loss = float('inf')
    best_config = None
    best_model = None
    
    for i, exp in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Testing: LR={exp['lr']}, Arch={exp['arch'][0]}, "
              f"Act={exp['act']}, Dropout={exp['drop']}, BS={exp['bs']}")
        
        # Construir modelo (similar a LSTM univariado)
        model = Sequential(name=f"LSTM_Multi_Exp{i}")
        units_list = exp['arch'][1]
        
        # Primera capa LSTM
        model.add(LSTM(units_list[0], activation=exp['act'], return_sequences=(len(units_list) > 1),
                      kernel_regularizer=l2(exp['l2']), input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(exp['drop']))
        
        # Capas LSTM adicionales
        for j, units in enumerate(units_list[1:]):
            return_seq = (j < len(units_list) - 2)
            model.add(LSTM(units, activation=exp['act'], return_sequences=return_seq,
                          kernel_regularizer=l2(exp['l2'])))
            model.add(Dropout(exp['drop']))
        
        # Capa de salida
        model.add(Dense(5, kernel_regularizer=l2(exp['l2'])))
        
        # Compilar
        optimizer = Adam(learning_rate=exp['lr']) if exp['opt'] == 'Adam' else RMSprop(learning_rate=exp['lr'])
        model.compile(optimizer=optimizer, loss=exp['loss'], metrics=['mae', 'mse'])
        
        # Entrenar
        history = model.fit(X_train, y_train,
                          validation_data=(X_val, y_val),
                          epochs=80,
                          batch_size=exp['bs'],
                          verbose=0)
        
        # Evaluar
        val_loss = history.history['val_loss'][-1]
        val_mae = history.history['val_mae'][-1]
        
        results.append({
            'experiment': i,
            'lr': exp['lr'],
            'architecture': exp['arch'][0],
            'activation': exp['act'],
            'dropout': exp['drop'],
            'batch_size': exp['bs'],
            'optimizer': exp['opt'],
            'loss_fn': exp['loss'],
            'l2_reg': exp['l2'],
            'val_loss': val_loss,
            'val_mae': val_mae
        })
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_config = exp
            best_model = model
            print(f"  → NUEVO MEJOR! Val Loss: {val_loss:.4f}")
        else:
            print(f"  → Val Loss: {val_loss:.4f}")
    
    results_df = pd.DataFrame(results)
    
    print("\n" + "="*80)
    print("RESULTADOS DE OPTIMIZACIÓN LSTM MULTIVARIADO")
    print("="*80)
    print(f"\nMejor configuración (Val Loss={best_val_loss:.4f}):")
    for key, value in best_config.items():
        if key != 'arch':
            print(f"  {key}: {value}")
        else:
            print(f"  architecture: {value[0]} - units: {value[1]}")
    
    return best_model, best_config, results_df


# ==================================================================================================
# VISUALIZATION FUNCTIONS
# ==================================================================================================

def visualize_tuning_results(results_df, model_name, output_dir='outputs'):
    """
    Crea visualizaciones comprehensivas de los resultados de tuning.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Learning Rate
    ax1 = fig.add_subplot(gs[0, 0])
    lr_grouped = results_df.groupby('lr')['val_loss'].mean()
    bars = ax1.bar(range(len(lr_grouped)), lr_grouped.values, color='skyblue', edgecolor='black')
    best_idx = lr_grouped.values.argmin()
    bars[best_idx].set_color('green')
    ax1.set_xticks(range(len(lr_grouped)))
    ax1.set_xticklabels([f'{x:.4f}' for x in lr_grouped.index])
    ax1.set_title('Learning Rate', fontweight='bold')
    ax1.set_ylabel('Val Loss Promedio')
    ax1.grid(True, alpha=0.3)
    
    # 2. Architecture (si existe)
    if 'architecture' in results_df.columns:
        ax2 = fig.add_subplot(gs[0, 1])
        arch_grouped = results_df.groupby('architecture')['val_loss'].mean()
        bars = ax2.bar(range(len(arch_grouped)), arch_grouped.values, color='coral', edgecolor='black')
        best_idx = arch_grouped.values.argmin()
        bars[best_idx].set_color('green')
        ax2.set_xticks(range(len(arch_grouped)))
        ax2.set_xticklabels(arch_grouped.index, rotation=45)
        ax2.set_title('Arquitectura', fontweight='bold')
        ax2.set_ylabel('Val Loss Promedio')
        ax2.grid(True, alpha=0.3)
    elif 'filters' in results_df.columns:
        ax2 = fig.add_subplot(gs[0, 1])
        # Para CNN, agrupar por filtros
        filter_grouped = results_df.groupby('filters')['val_loss'].mean()
        bars = ax2.bar(range(len(filter_grouped)), filter_grouped.values, color='coral', edgecolor='black')
        best_idx = filter_grouped.values.argmin()
        bars[best_idx].set_color('green')
        ax2.set_xticks(range(len(filter_grouped)))
        ax2.set_xticklabels([str(x) for x in filter_grouped.index], rotation=45, ha='right')
        ax2.set_title('Filtros CNN', fontweight='bold')
        ax2.set_ylabel('Val Loss Promedio')
        ax2.grid(True, alpha=0.3)
    
    # 3. Activation
    ax3 = fig.add_subplot(gs[0, 2])
    act_grouped = results_df.groupby('activation')['val_loss'].mean()
    bars = ax3.bar(range(len(act_grouped)), act_grouped.values, color='lightgreen', edgecolor='black')
    best_idx = act_grouped.values.argmin()
    bars[best_idx].set_color('green')
    ax3.set_xticks(range(len(act_grouped)))
    ax3.set_xticklabels(act_grouped.index)
    ax3.set_title('Función de Activación', fontweight='bold')
    ax3.set_ylabel('Val Loss Promedio')
    ax3.grid(True, alpha=0.3)
    
    # 4. Dropout
    ax4 = fig.add_subplot(gs[0, 3])
    drop_grouped = results_df.groupby('dropout')['val_loss'].mean()
    bars = ax4.bar(range(len(drop_grouped)), drop_grouped.values, color='gold', edgecolor='black')
    best_idx = drop_grouped.values.argmin()
    bars[best_idx].set_color('green')
    ax4.set_xticks(range(len(drop_grouped)))
    ax4.set_xticklabels(drop_grouped.index)
    ax4.set_title('Dropout Rate', fontweight='bold')
    ax4.set_ylabel('Val Loss Promedio')
    ax4.grid(True, alpha=0.3)
    
    # 5. Batch Size
    ax5 = fig.add_subplot(gs[1, 0])
    bs_grouped = results_df.groupby('batch_size')['val_loss'].mean()
    bars = ax5.bar(range(len(bs_grouped)), bs_grouped.values, color='plum', edgecolor='black')
    best_idx = bs_grouped.values.argmin()
    bars[best_idx].set_color('green')
    ax5.set_xticks(range(len(bs_grouped)))
    ax5.set_xticklabels(bs_grouped.index)
    ax5.set_title('Batch Size', fontweight='bold')
    ax5.set_ylabel('Val Loss Promedio')
    ax5.grid(True, alpha=0.3)
    
    # 6. Optimizer
    ax6 = fig.add_subplot(gs[1, 1])
    opt_grouped = results_df.groupby('optimizer')['val_loss'].mean()
    bars = ax6.bar(range(len(opt_grouped)), opt_grouped.values, color='lightblue', edgecolor='black')
    best_idx = opt_grouped.values.argmin()
    bars[best_idx].set_color('green')
    ax6.set_xticks(range(len(opt_grouped)))
    ax6.set_xticklabels(opt_grouped.index)
    ax6.set_title('Optimizador', fontweight='bold')
    ax6.set_ylabel('Val Loss Promedio')
    ax6.grid(True, alpha=0.3)
    
    # 7. Loss Function
    ax7 = fig.add_subplot(gs[1, 2])
    loss_grouped = results_df.groupby('loss_fn')['val_loss'].mean()
    bars = ax7.bar(range(len(loss_grouped)), loss_grouped.values, color='salmon', edgecolor='black')
    best_idx = loss_grouped.values.argmin()
    bars[best_idx].set_color('green')
    ax7.set_xticks(range(len(loss_grouped)))
    ax7.set_xticklabels([x.upper() for x in loss_grouped.index])
    ax7.set_title('Función de Pérdida', fontweight='bold')
    ax7.set_ylabel('Val Loss Promedio')
    ax7.grid(True, alpha=0.3)
    
    # 8. L2 Regularization
    ax8 = fig.add_subplot(gs[1, 3])
    l2_grouped = results_df.groupby('l2_reg')['val_loss'].mean()
    bars = ax8.bar(range(len(l2_grouped)), l2_grouped.values, color='khaki', edgecolor='black')
    best_idx = l2_grouped.values.argmin()
    bars[best_idx].set_color('green')
    ax8.set_xticks(range(len(l2_grouped)))
    ax8.set_xticklabels([f'{x:.4f}' for x in l2_grouped.index])
    ax8.set_title('Regularización L2', fontweight='bold')
    ax8.set_ylabel('Val Loss Promedio')
    ax8.grid(True, alpha=0.3)
    
    # 9. Top 10 experiments
    ax9 = fig.add_subplot(gs[2, :2])
    top10 = results_df.nsmallest(10, 'val_loss')
    colors = ['green' if i == 0 else 'lightblue' for i in range(len(top10))]
    ax9.barh(range(len(top10)), top10['val_loss'].values, color=colors, edgecolor='black')
    ax9.set_yticks(range(len(top10)))
    ax9.set_yticklabels([f"Exp {x}" for x in top10['experiment'].values])
    ax9.set_xlabel('Validation Loss')
    ax9.set_title('Top 10 Mejores Configuraciones', fontweight='bold', fontsize=12)
    ax9.invert_yaxis()
    ax9.grid(True, alpha=0.3, axis='x')
    
    # 10. Summary box
    ax10 = fig.add_subplot(gs[2, 2:])
    ax10.axis('off')
    best_exp = results_df.loc[results_df['val_loss'].idxmin()]
    summary_text = f"""
MEJOR CONFIGURACIÓN - {model_name}
{'='*50}

Experiment: #{int(best_exp['experiment'])}
Val Loss: {best_exp['val_loss']:.6f}
Val MAE: {best_exp['val_mae']:.6f}

Hyperparámetros:
  • Learning Rate: {best_exp['lr']}
  • {'Architecture' if 'architecture' in best_exp else 'Filters'}: {best_exp.get('architecture', best_exp.get('filters', 'N/A'))}
  • Activation: {best_exp['activation']}
  • Dropout: {best_exp['dropout']}
  • Batch Size: {int(best_exp['batch_size'])}
  • Optimizer: {best_exp['optimizer']}
  • Loss Function: {best_exp['loss_fn'].upper()}
  • L2 Regularization: {best_exp['l2_reg']}
"""
    ax10.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.suptitle(f'Resultados de Optimización - {model_name}', fontsize=16, fontweight='bold', y=0.995)
    
    filename = os.path.join(output_dir, f'tuning_{model_name.lower().replace(" ", "_")}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Visualización guardada: {filename}")


# ==================================================================================================
# PROJECTION FUNCTIONS
# ==================================================================================================

def project_baseline(model, last_window, scaler, steps=15):
    """
    Proyección iterativa para modelo baseline (regresión lineal).
    """
    projections = []
    current_window = last_window.copy()
    
    for _ in range(steps):
        # Flatten window para regresión lineal
        X_input = current_window.reshape(1, -1)
        # Predecir siguiente valor (tomar primer valor del horizonte de 5)
        pred = model.predict(X_input)[0][0]
        projections.append(pred)
        
        # Actualizar ventana (shift y agregar predicción)
        current_window = np.roll(current_window, -1)
        current_window[-1] = pred
    
    return np.array(projections)


def project_deep_learning(model, last_window, scaler, steps=15, is_multivariate=False):
    """
    Proyección iterativa para modelos de deep learning (LSTM/CNN).
    """
    projections = []
    current_window = last_window.copy()
    
    for _ in range(steps):
        # Reshape para modelo DL
        if is_multivariate:
            X_input = current_window.reshape(1, current_window.shape[0], current_window.shape[1])
        else:
            X_input = current_window.reshape(1, current_window.shape[0], 1)
        
        # Predecir siguiente valor
        pred = model.predict(X_input, verbose=0)[0][0]
        projections.append(pred)
        
        # Actualizar ventana
        if is_multivariate:
            # Para multivariado, actualizar solo la primera feature (Close)
            new_row = current_window[-1].copy()
            new_row[0] = pred
            current_window = np.vstack([current_window[1:], new_row])
        else:
            current_window = np.append(current_window[1:], pred)
    
    return np.array(projections)


def visualize_projections(df_test, predictions_dict, projections_dict, scaler_test, output_dir='outputs'):
    """
    Visualiza proyecciones de todos los modelos.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get last date from test set
    last_date_idx = len(df_test) - 1
    
    # Create figure with subplots for each model + combined
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.flatten()
    
    models = ['Baseline', 'LSTM Univariado', 'CNN Univariado', 'LSTM Multivariado']
    colors = ['gray', 'green', 'red', 'purple']
    
    for idx, (model_name, color) in enumerate(zip(models, colors)):
        ax = axes[idx]
        
        # Plot test data
        test_close = df_test['Close'].values
        ax.plot(range(len(test_close)), test_close, 'b-', label='Datos Reales (Test)', linewidth=2)
        
        # Plot predictions on test
        if model_name in predictions_dict:
            preds = predictions_dict[model_name]
            ax.plot(range(len(preds)), preds, f'{color}--', label=f'Predicciones {model_name}', 
                   linewidth=1.5, alpha=0.7)
        
        # Plot projections
        if model_name in projections_dict:
            proj = projections_dict[model_name]
            proj_start_idx = len(test_close)
            proj_indices = range(proj_start_idx, proj_start_idx + len(proj))
            ax.plot(proj_indices, proj, f'{color}-', label=f'Proyección {model_name}', 
                   linewidth=2, marker='o', markersize=4)
            # Add vertical line at projection start
            ax.axvline(x=proj_start_idx-1, color='black', linestyle=':', linewidth=2, alpha=0.5)
        
        ax.set_title(f'Proyección - {model_name}', fontweight='bold', fontsize=12)
        ax.set_xlabel('Índice Temporal')
        ax.set_ylabel('Precio Close (USD)')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    # Combined plot
    ax_combined = axes[4]
    ax_combined.plot(range(len(test_close)), test_close, 'b-', label='Datos Reales', linewidth=2.5)
    
    for model_name, color in zip(models, colors):
        if model_name in projections_dict:
            proj = projections_dict[model_name]
            proj_start_idx = len(test_close)
            proj_indices = range(proj_start_idx, proj_start_idx + len(proj))
            ax_combined.plot(proj_indices, proj, f'{color}-', label=f'{model_name}', 
                           linewidth=2, marker='o', markersize=3, alpha=0.7)
    
    ax_combined.axvline(x=len(test_close)-1, color='black', linestyle=':', linewidth=2, alpha=0.5)
    ax_combined.set_title('Comparación de Todas las Proyecciones', fontweight='bold', fontsize=14)
    ax_combined.set_xlabel('Índice Temporal')
    ax_combined.set_ylabel('Precio Close (USD)')
    ax_combined.legend(loc='best')
    ax_combined.grid(True, alpha=0.3)
    
    # Hide last subplot
    axes[5].axis('off')
    
    plt.suptitle('Proyecciones de Precios BNB - 15 Días Adelante', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    filename = os.path.join(output_dir, 'projections_all_models.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nVisualización de proyecciones guardada: {filename}")

