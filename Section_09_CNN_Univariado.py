"""
================================================================================
SECCIÓN 9/13: CNN UNIVARIADO
================================================================================
Entrena modelo CNN (Conv1D) con entrada univariada (solo Close).

Arquitectura: 2 capas Conv1D + MaxPooling + Dense
Hiperparámetros: LR=0.001, Epochs=100, Batch=32, No early stopping
Tiempo estimado: 8-12 minutos
Resultado esperado: R² ≈ 0.905
"""

print("=" * 80)
print("SECCIÓN 9/13: CNN UNIVARIADO")
print("=" * 80)

from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten

# ============================================================================
# 1. CONSTRUIR MODELO CNN
# ============================================================================
print("\n[1/3] Construyendo arquitectura CNN...")

model_cnn_univ = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(TIMESTEPS, 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(HORIZON)
], name='CNN_Univariado')

model_cnn_univ.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(f"✅ Modelo CNN Univariado construido")
model_cnn_univ.summary()

# ============================================================================
# 2. ENTRENAR MODELO
# ============================================================================
print("\n[2/3] Entrenando CNN Univariado (100 épocas, sin early stopping)...")
print("⏳ Esto tomará aproximadamente 8-12 minutos...")

history_cnn_univ = model_cnn_univ.fit(
    X_train_univ, y_train_univ,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_univ, y_val_univ),
    verbose=1
)

print(f"✅ Entrenamiento completado")

# ============================================================================
# 3. EVALUAR EN TEST SET
# ============================================================================
print("\n[3/3] Evaluando en test set...")

# Predicciones
y_pred_cnn_univ = model_cnn_univ.predict(X_test_univ)

# Usar promedio de predicciones multi-step
y_pred_cnn_univ_avg = y_pred_cnn_univ.mean(axis=1)
y_test_cnn_univ_avg = y_test_univ.mean(axis=1)

# Des-escalar
# Close es la primera columna (índice 0) en la lista de features_scale
close_idx = 0  # 'Close' es la primera columna

y_test_cnn_real = scaler_test.inverse_transform(
    np.concatenate([y_test_cnn_univ_avg.reshape(-1, 1)] + 
                   [np.zeros((len(y_test_cnn_univ_avg), 1)) for _ in range(4)], axis=1)
)[:, close_idx]

y_pred_cnn_real = scaler_test.inverse_transform(
    np.concatenate([y_pred_cnn_univ_avg.reshape(-1, 1)] + 
                   [np.zeros((len(y_pred_cnn_univ_avg), 1)) for _ in range(4)], axis=1)
)[:, close_idx]

# Métricas
mse_cnn_univ = mean_squared_error(y_test_cnn_real, y_pred_cnn_real)
mae_cnn_univ = mean_absolute_error(y_test_cnn_real, y_pred_cnn_real)
r2_cnn_univ = r2_score(y_test_cnn_real, y_pred_cnn_real)

print(f"\n📊 MÉTRICAS CNN UNIVARIADO:")
print(f"   MSE: {mse_cnn_univ:.2f}")
print(f"   MAE: {mae_cnn_univ:.2f}")
print(f"   R²:  {r2_cnn_univ:.4f}")

# ============================================================================
# 4. VISUALIZACIÓN
# ============================================================================
print("\n📈 Generando visualización...")

plt.figure(figsize=(14, 6))
plt.plot(y_test_cnn_real, label='Valores Reales', color='blue', linewidth=2)
plt.plot(y_pred_cnn_real, label='Predicciones CNN', color='red', linewidth=2, alpha=0.7)
plt.title('Predicciones vs Valores Reales - CNN Univariado', fontsize=14, fontweight='bold')
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
print("SECCIÓN 9 COMPLETADA ✅")
print("=" * 80)
print(f"\n🎯 Modelo CNN Univariado entrenado:")
print(f"   • Arquitectura: 2 capas Conv1D + MaxPooling")
print(f"   • R²: {r2_cnn_univ:.4f}")
print(f"   • MSE: {mse_cnn_univ:.2f}")
print(f"   • MAE: {mae_cnn_univ:.2f} USD")
print(f"\n📊 Comparación con modelos anteriores:")
print(f"   • Baseline R²: {r2_baseline:.4f}")
print(f"   • LSTM R²:     {r2_lstm_univ:.4f}")
print(f"   • CNN R²:      {r2_cnn_univ:.4f}")
print(f"\n🎯 Variables en memoria:")
print("   • model_cnn_univ, history_cnn_univ")
print("   • y_pred_cnn_real, y_test_cnn_real")
print("   • mse_cnn_univ, mae_cnn_univ, r2_cnn_univ")
print("\n➡️  Continuar con Sección 10: LSTM Multivariado")
print("=" * 80)
