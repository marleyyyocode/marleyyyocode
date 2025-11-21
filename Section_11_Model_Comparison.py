# ================================================================================
# SECCIÓN 11/13: COMPARACIÓN DE TODOS LOS MODELOS
# ================================================================================
# 
# REQUISITOS: Haber ejecutado las secciones 1-10 previamente
# TIEMPO ESTIMADO: 10-20 segundos
# 
# OUTPUTS:
# - Tabla comparativa de métricas (MSE, MAE, R²)
# - Gráfica combinada de todos los modelos vs valores reales
# - Identificación del mejor modelo
# ================================================================================

print("\n" + "="*80)
print("SECCIÓN 11/13: COMPARACIÓN DE TODOS LOS MODELOS")
print("="*80)

# Importaciones necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ================================================================================
# PASO 1: Recopilar Métricas de Todos los Modelos
# ================================================================================
print("\n[1/3] Recopilando métricas de todos los modelos...")

# Crear DataFrame con métricas
metrics_comparison = pd.DataFrame({
    'Modelo': ['Baseline (Linear Regression)', 'LSTM Univariado', 'CNN Univariado', 'LSTM Multivariado'],
    'MSE': [mse_baseline, mse_lstm_univ, mse_cnn_univ, mse_lstm_multi],
    'MAE': [mae_baseline, mae_lstm_univ, mae_cnn_univ, mae_lstm_multi],
    'R²': [r2_baseline, r2_lstm_univ, r2_cnn_univ, r2_lstm_multi]
})

# Ordenar por R² descendente
metrics_comparison = metrics_comparison.sort_values('R²', ascending=False).reset_index(drop=True)

print("\n" + "="*80)
print("MÉTRICAS COMPARATIVAS - TEST SET")
print("="*80)
print(metrics_comparison.to_string(index=False))
print("="*80)

# Identificar mejor modelo
best_model = metrics_comparison.iloc[0]['Modelo']
best_r2 = metrics_comparison.iloc[0]['R²']

print(f"\n🏆 MEJOR MODELO: {best_model} (R² = {best_r2:.4f})")

# ================================================================================
# PASO 2: Crear Gráfica Comparativa
# ================================================================================
print("\n[2/3] Generando gráfica comparativa...")

# Crear figura
plt.figure(figsize=(15, 8))

# Graficar valores reales
plt.plot(y_test_real[:, 0], label='Valores Reales', color='blue', linewidth=2, alpha=0.8)

# Graficar predicciones de cada modelo (usando variables _real de secciones 7-10)
plt.plot(y_pred_real[:, 0], label='Baseline', color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
plt.plot(y_pred_lstm_real[:, 0], label='LSTM Univariado', color='green', linestyle='-', linewidth=1.5, alpha=0.7)
plt.plot(y_pred_cnn_real[:, 0], label='CNN Univariado', color='red', linestyle='-', linewidth=1.5, alpha=0.7)
plt.plot(y_pred_lstm_multi_real[:, 0], label='LSTM Multivariado', color='purple', linestyle='-', linewidth=1.5, alpha=0.7)

# Configurar gráfica
plt.title('Comparación de Todos los Modelos - Test Set', fontsize=16, fontweight='bold')
plt.xlabel('Índice', fontsize=12)
plt.ylabel('Precio Close (USD)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

print("✅ Gráfica comparativa generada")
plt.show()

# ================================================================================
# PASO 3: Análisis de Resultados
# ================================================================================
print("\n[3/3] Análisis de resultados...")

print("\n" + "="*80)
print("ANÁLISIS DE RESULTADOS")
print("="*80)

# Calcular diferencias entre modelos
baseline_r2 = metrics_comparison[metrics_comparison['Modelo'] == 'Baseline (Linear Regression)']['R²'].values[0]
best_dl_model = metrics_comparison[metrics_comparison['Modelo'] != 'Baseline (Linear Regression)'].iloc[0]
best_dl_r2 = best_dl_model['R²']
best_dl_name = best_dl_model['Modelo']

print(f"\n📊 Baseline R²: {baseline_r2:.4f}")
print(f"📊 Mejor modelo DL: {best_dl_name} (R² = {best_dl_r2:.4f})")
print(f"📊 Diferencia: {abs(baseline_r2 - best_dl_r2):.4f}")

if baseline_r2 > best_dl_r2:
    print(f"\n💡 El Baseline supera a los modelos de deep learning por {(baseline_r2 - best_dl_r2)*100:.2f}%")
    print("   Esto sugiere que la relación en los datos es mayormente lineal.")
else:
    print(f"\n💡 Los modelos de deep learning superan al Baseline por {(best_dl_r2 - baseline_r2)*100:.2f}%")
    print("   Esto indica que los modelos DL capturan patrones no lineales efectivamente.")

# Análisis por modelo
print("\n" + "="*80)
print("OBSERVACIONES POR MODELO:")
print("="*80)

for idx, row in metrics_comparison.iterrows():
    modelo = row['Modelo']
    r2 = row['R²']
    mse = row['MSE']
    
    print(f"\n{idx+1}. {modelo}:")
    print(f"   - R²: {r2:.4f} ({'Excelente' if r2 > 0.95 else 'Muy Bueno' if r2 > 0.90 else 'Bueno' if r2 > 0.80 else 'Aceptable' if r2 > 0.70 else 'Necesita Mejora'})")
    print(f"   - MSE: {mse:.2f}")
    
    if 'Baseline' in modelo and r2 > 0.95:
        print(f"   ✓ Modelo de referencia con excelente rendimiento")
    elif 'LSTM Univariado' in modelo and r2 > 0.90:
        print(f"   ✓ Mejor modelo de deep learning, captura bien las tendencias")
    elif 'CNN' in modelo and r2 > 0.85:
        print(f"   ✓ Buen rendimiento, captura patrones locales")
    elif 'Multivariado' in modelo and r2 < 0.70:
        print(f"   ! Rendimiento bajo, las variables adicionales no ayudan significativamente")

print("\n" + "="*80)
print("✅ SECCIÓN 11 COMPLETADA")
print("="*80)
print("\nPróxima sección: Proyecciones a 15 días (Section_12_Projections_15_Days.py)")
