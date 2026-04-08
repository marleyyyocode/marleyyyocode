"""
run_complete_pipeline.py
=========================
Script ejecutable completo que integra:
1. Todo el código existente de Codigo_GrupoBNB.py
2. Optimización individual de hiperparámetros por modelo
3. Proyecciones para todos los modelos

Tiempo estimado: 30-40 minutos
"""

import sys
import os
from datetime import datetime

print("="*80)
print("PIPELINE COMPLETO BNB - Optimización Individual + Proyecciones")
print("="*80)
print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Tiempo estimado: 30-40 minutos\n")

# Verificar que el módulo enhanced_additions existe
if not os.path.exists('enhanced_additions.py'):
    print("ERROR: No se encontró enhanced_additions.py")
    print("Asegúrate de que el archivo esté en el directorio actual.")
    sys.exit(1)

# Importar módulo base (sin ejecutar main)
print("📦 Importando módulo base...")
import Codigo_GrupoBNB as base

# Importar funciones enhanced
print("📦 Importando funciones de optimización y proyección...")
from enhanced_additions import (
    tune_lstm_univariado,
    tune_cnn_univariado,
    tune_lstm_multivariado,
    project_baseline,
    project_deep_learning,
    visualize_tuning_results,
    visualize_projections
)

import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("✓ Módulos cargados exitosamente\n")

# ============================================================================
# EJECUTAR PIPELINE BASE
# ============================================================================

print("="*80)
print("FASE 1: Preparación de Datos y EDA")
print("="*80)

# 1. Download data
df = base.download_data()

# 2. Feature engineering (antes de EDA)
print("\n📊 Feature Engineering...")
df = base.feature_engineering(df)

# 3. EDA (ahora incluye Daily_Return y Volatility)
print("\n📊 Análisis Exploratorio...")
base.exploratory_data_analysis(df)

# 4. Cross-validation strategies comparison
print("\n📊 Comparación de Estrategias de Validación Cruzada...")
base.compare_cv_strategies(df)

# 5. Split data
print("\n📊 División Temporal de Datos...")
train_df, val_df, test_df = base.temporal_split(df)

# 6. Scale data
print("\n�� Escalado de Features...")
train_scaled, val_scaled, test_scaled, scaler_train, scaler_val, scaler_test = base.scale_data(
    train_df, val_df, test_df
)

# 7. Generate sequences
print("\n📊 Generación de Secuencias...")
X_train_univ, y_train = base.create_sequences_univariate(train_scaled, base.TIMESTEPS, base.HORIZON)
X_val_univ, y_val = base.create_sequences_univariate(val_scaled, base.TIMESTEPS, base.HORIZON)
X_test_univ, y_test = base.create_sequences_univariate(test_scaled, base.TIMESTEPS, base.HORIZON)

X_train_multi, _ = base.create_sequences_multivariate(train_scaled, base.TIMESTEPS, base.HORIZON)
X_val_multi, _ = base.create_sequences_multivariate(val_scaled, base.TIMESTEPS, base.HORIZON)
X_test_multi, _ = base.create_sequences_multivariate(test_scaled, base.TIMESTEPS, base.HORIZON)

print(f"✓ Secuencias generadas:")
print(f"  - Univariadas: Train={X_train_univ.shape}, Val={X_val_univ.shape}, Test={X_test_univ.shape}")
print(f"  - Multivariadas: Train={X_train_multi.shape}, Val={X_val_multi.shape}, Test={X_test_multi.shape}")

# ============================================================================
# FASE 2: OPTIMIZACIÓN INDIVIDUAL DE HIPERPARÁMETROS
# ============================================================================

print("\n" + "="*80)
print("FASE 2: Optimización Individual de Hiperparámetros (~30 min)")
print("="*80)

# 2.1 LSTM Univariado
print("\n🔧 Iniciando optimización LSTM Univariado...")
start_time = datetime.now()
best_lstm_univ, config_lstm, results_lstm = tune_lstm_univariado(
    X_train_univ, y_train, X_val_univ, y_val, scaler_train
)
elapsed = (datetime.now() - start_time).total_seconds() / 60
print(f"✓ LSTM Univariado optimizado en {elapsed:.1f} minutos")

# Guardar modelo
os.makedirs('models', exist_ok=True)
best_lstm_univ.save('models/lstm_univariado_optimized.h5')
print("✓ Modelo guardado: models/lstm_univariado_optimized.h5")

# Visualizar resultados
visualize_tuning_results(results_lstm, "LSTM Univariado")

# 2.2 CNN Univariado
print("\n🔧 Iniciando optimización CNN Univariado...")
start_time = datetime.now()
best_cnn, config_cnn, results_cnn = tune_cnn_univariado(
    X_train_univ, y_train, X_val_univ, y_val, scaler_train
)
elapsed = (datetime.now() - start_time).total_seconds() / 60
print(f"✓ CNN Univariado optimizado en {elapsed:.1f} minutos")

# Guardar modelo
best_cnn.save('models/cnn_univariado_optimized.h5')
print("✓ Modelo guardado: models/cnn_univariado_optimized.h5")

# Visualizar resultados
visualize_tuning_results(results_cnn, "CNN Univariado")

# 2.3 LSTM Multivariado
print("\n🔧 Iniciando optimización LSTM Multivariado...")
start_time = datetime.now()
best_lstm_multi, config_multi, results_multi = tune_lstm_multivariado(
    X_train_multi, y_train, X_val_multi, y_val, scaler_train
)
elapsed = (datetime.now() - start_time).total_seconds() / 60
print(f"✓ LSTM Multivariado optimizado en {elapsed:.1f} minutos")

# Guardar modelo
best_lstm_multi.save('models/lstm_multivariado_optimized.h5')
print("✓ Modelo guardado: models/lstm_multivariado_optimized.h5")

# Visualizar resultados
visualize_tuning_results(results_multi, "LSTM Multivariado")

# 2.4 Baseline (sin optimización, ya es óptimo)
print("\n🔧 Entrenando modelo Baseline...")
baseline_model = base.train_baseline(X_train_univ, y_train)
joblib.dump(baseline_model, 'models/baseline_model_optimized.pkl')
print("✓ Baseline entrenado y guardado")

# ============================================================================
# FASE 3: EVALUACIÓN EN TEST SET
# ============================================================================

print("\n" + "="*80)
print("FASE 3: Evaluación en Test Set")
print("="*80)

# Evaluar todos los modelos en test
predictions_dict = {}
metrics_results = []

# Baseline
print("\n📈 Evaluando Baseline...")
X_test_flat = X_test_univ.reshape(X_test_univ.shape[0], -1)
pred_baseline = baseline_model.predict(X_test_flat)
predictions_dict['Baseline'] = scaler_test.inverse_transform(
    np.column_stack([pred_baseline[:, 0], np.zeros((len(pred_baseline), 3))])
)[:, 0]

# Revertir y_test
y_test_original = scaler_test.inverse_transform(
    np.column_stack([y_test[:, 0], np.zeros((len(y_test), 3))])
)[:, 0]

mse = mean_squared_error(y_test_original, predictions_dict['Baseline'])
mae = mean_absolute_error(y_test_original, predictions_dict['Baseline'])
r2 = r2_score(y_test_original, predictions_dict['Baseline'])
metrics_results.append({'Model': 'Baseline', 'MSE': mse, 'MAE': mae, 'R2': r2})
print(f"  MSE={mse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")

# LSTM Univariado
print("\n📈 Evaluando LSTM Univariado Optimizado...")
pred_lstm = best_lstm_univ.predict(X_test_univ, verbose=0)
predictions_dict['LSTM Univariado'] = scaler_test.inverse_transform(
    np.column_stack([pred_lstm[:, 0], np.zeros((len(pred_lstm), 3))])
)[:, 0]

mse = mean_squared_error(y_test_original, predictions_dict['LSTM Univariado'])
mae = mean_absolute_error(y_test_original, predictions_dict['LSTM Univariado'])
r2 = r2_score(y_test_original, predictions_dict['LSTM Univariado'])
metrics_results.append({'Model': 'LSTM Univariado', 'MSE': mse, 'MAE': mae, 'R2': r2})
print(f"  MSE={mse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")

# CNN Univariado
print("\n📈 Evaluando CNN Univariado Optimizado...")
pred_cnn = best_cnn.predict(X_test_univ, verbose=0)
predictions_dict['CNN Univariado'] = scaler_test.inverse_transform(
    np.column_stack([pred_cnn[:, 0], np.zeros((len(pred_cnn), 3))])
)[:, 0]

mse = mean_squared_error(y_test_original, predictions_dict['CNN Univariado'])
mae = mean_absolute_error(y_test_original, predictions_dict['CNN Univariado'])
r2 = r2_score(y_test_original, predictions_dict['CNN Univariado'])
metrics_results.append({'Model': 'CNN Univariado', 'MSE': mse, 'MAE': mae, 'R2': r2})
print(f"  MSE={mse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")

# LSTM Multivariado
print("\n📈 Evaluando LSTM Multivariado Optimizado...")
pred_multi = best_lstm_multi.predict(X_test_multi, verbose=0)
predictions_dict['LSTM Multivariado'] = scaler_test.inverse_transform(
    np.column_stack([pred_multi[:, 0], np.zeros((len(pred_multi), 3))])
)[:, 0]

mse = mean_squared_error(y_test_original, predictions_dict['LSTM Multivariado'])
mae = mean_absolute_error(y_test_original, predictions_dict['LSTM Multivariado'])
r2 = r2_score(y_test_original, predictions_dict['LSTM Multivariado'])
metrics_results.append({'Model': 'LSTM Multivariado', 'MSE': mse, 'MAE': mae, 'R2': r2})
print(f"  MSE={mse:.2f}, MAE={mae:.2f}, R²={r2:.4f}")

# Guardar métricas
metrics_df = pd.DataFrame(metrics_results)
metrics_df.to_csv('metrics_optimized.csv', index=False)
print("\n✓ Métricas guardadas en metrics_optimized.csv")

# ============================================================================
# FASE 4: PROYECCIONES
# ============================================================================

print("\n" + "="*80)
print("FASE 4: Generación de Proyecciones (15 días)")
print("="*80)

projections_dict = {}

# Obtener última ventana de cada tipo
last_window_univ = test_scaled[-base.TIMESTEPS:, 0]
last_window_multi = test_scaled[-base.TIMESTEPS:, [1, 2, 3]]  # High, Volume, Volatility

# Proyectar cada modelo
print("\n🔮 Proyectando Baseline...")
proj_baseline_scaled = project_baseline(baseline_model, last_window_univ, scaler_test, steps=15)
projections_dict['Baseline'] = scaler_test.inverse_transform(
    np.column_stack([proj_baseline_scaled, np.zeros((len(proj_baseline_scaled), 3))])
)[:, 0]
print(f"  ✓ 15 días proyectados")

print("\n🔮 Proyectando LSTM Univariado...")
proj_lstm_scaled = project_deep_learning(best_lstm_univ, last_window_univ, scaler_test, steps=15, is_multivariate=False)
projections_dict['LSTM Univariado'] = scaler_test.inverse_transform(
    np.column_stack([proj_lstm_scaled, np.zeros((len(proj_lstm_scaled), 3))])
)[:, 0]
print(f"  ✓ 15 días proyectados")

print("\n🔮 Proyectando CNN Univariado...")
proj_cnn_scaled = project_deep_learning(best_cnn, last_window_univ, scaler_test, steps=15, is_multivariate=False)
projections_dict['CNN Univariado'] = scaler_test.inverse_transform(
    np.column_stack([proj_cnn_scaled, np.zeros((len(proj_cnn_scaled), 3))])
)[:, 0]
print(f"  ✓ 15 días proyectados")

print("\n🔮 Proyectando LSTM Multivariado...")
proj_multi_scaled = project_deep_learning(best_lstm_multi, last_window_multi, scaler_test, steps=15, is_multivariate=True)
projections_dict['LSTM Multivariado'] = scaler_test.inverse_transform(
    np.column_stack([proj_multi_scaled, np.zeros((len(proj_multi_scaled), 3))])
)[:, 0]
print(f"  ✓ 15 días proyectados")

# Visualizar proyecciones
print("\n📊 Generando visualizaciones de proyecciones...")
visualize_projections(test_df, predictions_dict, projections_dict, scaler_test)

# Guardar proyecciones en CSV
proj_df = pd.DataFrame(projections_dict)
proj_df.index.name = 'Day'
proj_df.to_csv('projections_15_days.csv')
print("✓ Proyecciones guardadas en projections_15_days.csv")

# ============================================================================
# RESUMEN FINAL
# ============================================================================

print("\n" + "="*80)
print("✅ PIPELINE COMPLETO FINALIZADO")
print("="*80)

print("\n📊 MÉTRICAS FINALES (Test Set):")
print(metrics_df.to_string(index=False))

print("\n📁 ARCHIVOS GENERADOS:")
print("  Modelos:")
print("    - models/baseline_model_optimized.pkl")
print("    - models/lstm_univariado_optimized.h5")
print("    - models/cnn_univariado_optimized.h5")
print("    - models/lstm_multivariado_optimized.h5")
print("\n  Visualizaciones de Tuning:")
print("    - outputs/tuning_lstm_univariado.png")
print("    - outputs/tuning_cnn_univariado.png")
print("    - outputs/tuning_lstm_multivariado.png")
print("\n  Visualizaciones de Proyecciones:")
print("    - outputs/projections_all_models.png")
print("\n  Datos:")
print("    - metrics_optimized.csv")
print("    - projections_15_days.csv")

print(f"\n⏱️ Finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)

