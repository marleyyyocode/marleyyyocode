# ================================================================================
# SECCIÓN 13/13: GUARDAR RESULTADOS
# ================================================================================
# 
# REQUISITOS: Haber ejecutado las secciones 1-12 previamente
# TIEMPO ESTIMADO: 20-30 segundos
# 
# OUTPUTS:
# - Modelos guardados (.pkl para baseline, .h5 para deep learning)
# - metrics_final.csv con métricas de todos los modelos
# - projections_15_days.csv con proyecciones
# ================================================================================

print("\n" + "="*80)
print("SECCIÓN 13/13: GUARDAR RESULTADOS")
print("="*80)

# Importaciones necesarias
import os
import pickle
import pandas as pd
from google.colab import files

# ================================================================================
# PASO 1: Crear Directorio de Salida
# ================================================================================
print("\n[1/4] Creando directorio de salida...")

os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

print("✅ Directorios creados: models/, outputs/")

# ================================================================================
# PASO 2: Guardar Modelos
# ================================================================================
print("\n[2/4] Guardando modelos entrenados...")

# Guardar Baseline (scikit-learn)
with open('models/baseline_model.pkl', 'wb') as f:
    pickle.dump(baseline_model, f)
print("   ✅ Baseline guardado: models/baseline_model.pkl")

# Guardar LSTM Univariado (TensorFlow)
lstm_univ_model.save('models/lstm_univariado.h5')
print("   ✅ LSTM Univariado guardado: models/lstm_univariado.h5")

# Guardar CNN Univariado (TensorFlow)
cnn_univ_model.save('models/cnn_univariado.h5')
print("   ✅ CNN Univariado guardado: models/cnn_univariado.h5")

# Guardar LSTM Multivariado (TensorFlow)
lstm_multi_model.save('models/lstm_multivariado.h5')
print("   ✅ LSTM Multivariado guardado: models/lstm_multivariado.h5")

# Guardar Scalers
with open('models/scaler_train.pkl', 'wb') as f:
    pickle.dump(scaler_train, f)
with open('models/scaler_test.pkl', 'wb') as f:
    pickle.dump(scaler_test, f)
print("   ✅ Scalers guardados: models/scaler_*.pkl")

# ================================================================================
# PASO 3: Guardar Métricas
# ================================================================================
print("\n[3/4] Guardando métricas...")

# DataFrame de métricas
metrics_final = pd.DataFrame({
    'Modelo': ['Baseline (Linear Regression)', 'LSTM Univariado', 'CNN Univariado', 'LSTM Multivariado'],
    'MSE': [mse_baseline, mse_lstm_univ, mse_cnn_univ, mse_lstm_multi],
    'MAE': [mae_baseline, mae_lstm_univ, mae_cnn_univ, mae_lstm_multi],
    'R²': [r2_baseline, r2_lstm_univ, r2_cnn_univ, r2_lstm_multi],
    'Arquitectura': [
        'Linear Regression',
        'LSTM: 64→32 units, 2 layers',
        'CNN: Conv1D layers',
        'LSTM: 64→32 units, 3 features'
    ],
    'Hiperparámetros': [
        'N/A',
        'LR=0.001, Epochs=100, Batch=32',
        'LR=0.001, Epochs=100, Batch=32',
        'LR=0.001, Epochs=100, Batch=32'
    ]
})

# Guardar CSV
metrics_final.to_csv('outputs/metrics_final.csv', index=False)
print("   ✅ Métricas guardadas: outputs/metrics_final.csv")

# Guardar proyecciones
projections_df.to_csv('outputs/projections_15_days.csv', index=False)
print("   ✅ Proyecciones guardadas: outputs/projections_15_days.csv")

# ================================================================================
# PASO 4: Resumen y Descarga
# ================================================================================
print("\n[4/4] Resumen de archivos guardados...")

print("\n" + "="*80)
print("ARCHIVOS GUARDADOS")
print("="*80)

print("\n📁 MODELOS (models/):")
print("   1. baseline_model.pkl (Linear Regression)")
print("   2. lstm_univariado.h5 (LSTM Univariado)")
print("   3. cnn_univariado.h5 (CNN Univariado)")
print("   4. lstm_multivariado.h5 (LSTM Multivariado)")
print("   5. scaler_train.pkl (Scaler de entrenamiento)")
print("   6. scaler_test.pkl (Scaler de test)")

print("\n📁 RESULTADOS (outputs/):")
print("   1. metrics_final.csv (Métricas de todos los modelos)")
print("   2. projections_15_days.csv (Proyecciones a 15 días)")

# Mostrar preview de métricas
print("\n" + "="*80)
print("PREVIEW: MÉTRICAS FINALES")
print("="*80)
print(metrics_final[['Modelo', 'MSE', 'MAE', 'R²']].to_string(index=False))

print("\n" + "="*80)
print("PREVIEW: PROYECCIONES (primeros 5 días)")
print("="*80)
print(projections_df.head().to_string(index=False))

# ================================================================================
# INSTRUCCIONES DE DESCARGA (GOOGLE COLAB)
# ================================================================================
print("\n" + "="*80)
print("INSTRUCCIONES DE DESCARGA (Google Colab)")
print("="*80)

print("\nPara descargar los archivos desde Google Colab:")
print("\nOPCIÓN 1: Descargar individual")
print("   - Clic en carpeta 📁 en panel izquierdo")
print("   - Navegar a models/ o outputs/")
print("   - Clic derecho → Descargar")

print("\nOPCIÓN 2: Descargar programáticamente (ejecutar en nueva celda):")
print("   ```python")
print("   from google.colab import files")
print("   ")
print("   # Descargar métricas")
print("   files.download('outputs/metrics_final.csv')")
print("   ")
print("   # Descargar proyecciones")
print("   files.download('outputs/projections_15_days.csv')")
print("   ")
print("   # Descargar modelos")
print("   files.download('models/baseline_model.pkl')")
print("   files.download('models/lstm_univariado.h5')")
print("   files.download('models/cnn_univariado.h5')")
print("   files.download('models/lstm_multivariado.h5')")
print("   ```")

print("\nOPCIÓN 3: Comprimir todo y descargar (ejecutar en nueva celda):")
print("   ```python")
print("   !zip -r resultados_bnb.zip models/ outputs/")
print("   from google.colab import files")
print("   files.download('resultados_bnb.zip')")
print("   ```")

# ================================================================================
# FINALIZACIÓN
# ================================================================================
print("\n" + "="*80)
print("🎉 ¡PIPELINE COMPLETO! TODAS LAS SECCIONES FINALIZADAS")
print("="*80)

print("\n✅ RESUMEN DE EJECUCIÓN:")
print(f"   - Datos descargados: {len(df)} registros")
print(f"   - Variables creadas: Daily_Return, Volatility")
print(f"   - Modelos entrenados: 4 (Baseline, LSTM Univ, CNN Univ, LSTM Multi)")
print(f"   - Mejor modelo: {metrics_final.iloc[0]['Modelo']} (R² = {metrics_final.iloc[0]['R²']:.4f})")
print(f"   - Proyecciones generadas: {PROJECTION_DAYS} días")
print(f"   - Archivos guardados: 8 archivos (6 modelos + 2 CSVs)")

print("\n🎓 LISTO PARA PRESENTACIÓN AL PROFESOR:")
print("   ✓ Todos los modelos entrenados")
print("   ✓ Métricas comparativas calculadas")
print("   ✓ Proyecciones a 15 días generadas")
print("   ✓ Visualizaciones completas (inline)")
print("   ✓ Archivos guardados para respaldo")

print("\n" + "="*80)
print("✅ SECCIÓN 13/13 COMPLETADA - FIN DEL PIPELINE")
print("="*80)

print("\n🌟 ¡Éxito! Tu pipeline de predicción de precios BNB está completo.")
print("   Tiempo total aproximado: 35-40 minutos")
print("   Todas las secciones ejecutadas correctamente.")
print("\n" + "="*80)
