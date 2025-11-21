"""
================================================================================
SECCIÓN 7/13: MODELO BASELINE (LINEAR REGRESSION)
================================================================================
Entrena modelo de referencia (Regresión Lineal) para comparar con deep learning.

Tiempo estimado: 10 segundos
Resultado esperado: R² ≈ 0.967
"""

print("=" * 80)
print("SECCIÓN 7/13: MODELO BASELINE (LINEAR REGRESSION)")
print("=" * 80)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================================
# 1. PREPARAR DATOS PARA REGRESIÓN LINEAL
# ============================================================================
print("\n[1/3] Preparando datos...")

# Usar promedio de los HORIZON valores como target
y_train_baseline = y_train_univ.mean(axis=1)
y_test_baseline = y_test_univ.mean(axis=1)

# Reshape X para sklearn (samples, features)
X_train_baseline = X_train_univ.reshape(X_train_univ.shape[0], -1)
X_test_baseline = X_test_univ.reshape(X_test_univ.shape[0], -1)

print(f"✅ X_train: {X_train_baseline.shape}")
print(f"✅ X_test:  {X_test_baseline.shape}")

# ============================================================================
# 2. ENTRENAR MODELO BASELINE
# ============================================================================
print("\n[2/3] Entrenando Linear Regression...")

baseline_model = LinearRegression()
baseline_model.fit(X_train_baseline, y_train_baseline)

print(f"✅ Modelo entrenado")
print(f"   Coeficientes: {baseline_model.coef_.shape}")
print(f"   Intercepto: {baseline_model.intercept_:.4f}")

# ============================================================================
# 3. EVALUAR EN TEST SET
# ============================================================================
print("\n[3/3] Evaluando en test set...")

# Predicciones
y_pred_baseline = baseline_model.predict(X_test_baseline)

# Des-escalar predicciones y valores reales
# Close es la primera columna (índice 0) en la lista de features_scale
close_idx = 0  # 'Close' es la primera columna

y_test_real = scaler_test.inverse_transform(
    np.concatenate([y_test_baseline.reshape(-1, 1)] + 
                   [np.zeros((len(y_test_baseline), 1)) for _ in range(4)], axis=1)
)[:, close_idx]

y_pred_real = scaler_test.inverse_transform(
    np.concatenate([y_pred_baseline.reshape(-1, 1)] + 
                   [np.zeros((len(y_pred_baseline), 1)) for _ in range(4)], axis=1)
)[:, close_idx]

# Calcular métricas
mse_baseline = mean_squared_error(y_test_real, y_pred_real)
mae_baseline = mean_absolute_error(y_test_real, y_pred_real)
r2_baseline = r2_score(y_test_real, y_pred_real)

print(f"\n📊 MÉTRICAS BASELINE:")
print(f"   MSE: {mse_baseline:.2f}")
print(f"   MAE: {mae_baseline:.2f}")
print(f"   R²:  {r2_baseline:.4f}")

# ============================================================================
# 4. VISUALIZACIÓN
# ============================================================================
print("\n📈 Generando visualización...")

plt.figure(figsize=(14, 6))
plt.plot(y_test_real, label='Valores Reales', color='blue', linewidth=2)
plt.plot(y_pred_real, label='Predicciones Baseline', color='orange', linewidth=2, alpha=0.7)
plt.title('Predicciones vs Valores Reales - Baseline (Linear Regression)', fontsize=14, fontweight='bold')
plt.xlabel('Índice / Fecha', fontsize=12)
plt.ylabel('Precio Close (USD)', fontsize=12)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# RESUMEN
# ============================================================================
print("\n" + "=" * 80)
print("SECCIÓN 7 COMPLETADA ✅")
print("=" * 80)
print(f"\n🎯 Modelo Baseline entrenado:")
print(f"   • Algoritmo: Linear Regression")
print(f"   • R²: {r2_baseline:.4f} (muy bueno)")
print(f"   • MSE: {mse_baseline:.2f}")
print(f"   • MAE: {mae_baseline:.2f} USD")
print(f"\n📝 Este es el modelo de referencia para comparar con LSTM y CNN")
print(f"\n🎯 Variables en memoria:")
print("   • baseline_model")
print("   • y_pred_baseline, y_test_real, y_pred_real")
print("   • mse_baseline, mae_baseline, r2_baseline")
print("\n➡️  Continuar con Sección 8: LSTM Univariado")
print("=" * 80)
