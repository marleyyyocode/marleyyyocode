"""
================================================================================
SECCIÓN 10/13: LSTM MULTIVARIADO
================================================================================
Entrena modelo LSTM con entrada multivariada (High, Volume, Volatility).

Arquitectura: 2 capas LSTM (64→32 unidades)
Entrada: 3 features (High, Volume, Volatility)
Output: Close
Hiperparámetros: LR=0.001, Epochs=100, Batch=32, No early stopping
Tiempo estimado: 8-12 minutos
Resultado esperado: R² ≈ 0.483
"""

print("=" * 80)
print("SECCIÓN 10/13: LSTM MULTIVARIADO")
print("=" * 80)

# ============================================================================
# 1. CONSTRUIR MODELO LSTM MULTIVARIADO
# ============================================================================
print("\n[1/3] Construyendo arquitectura LSTM Multivariada...")

model_lstm_multi = Sequential([
    LSTM(64, return_sequences=True, input_shape=(TIMESTEPS, 3)),  # 3 features
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(HORIZON)
], name='LSTM_Multivariado')

model_lstm_multi.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(f"✅ Modelo LSTM Multivariado construido")
model_lstm_multi.summary()

# ============================================================================
# 2. ENTRENAR MODELO
# ============================================================================
print("\n[2/3] Entrenando LSTM Multivariado (100 épocas, sin early stopping)...")
print("⏳ Esto tomará aproximadamente 8-12 minutos...")

history_lstm_multi = model_lstm_multi.fit(
    X_train_multi, y_train_multi,
    epochs=100,
    batch_size=32,
    validation_data=(X_val_multi, y_val_multi),
    verbose=1
)

print(f"✅ Entrenamiento completado")

# ============================================================================
# 3. EVALUAR EN TEST SET
# ============================================================================
print("\n[3/3] Evaluando en test set...")

# Predicciones
y_pred_lstm_multi = model_lstm_multi.predict(X_test_multi)

# Usar promedio de predicciones multi-step
y_pred_lstm_multi_avg = y_pred_lstm_multi.mean(axis=1)
y_test_lstm_multi_avg = y_test_multi.mean(axis=1)

# Des-escalar
close_idx = df_test_scaled.columns.get_loc(('Close', 'BNB-USD'))

y_test_lstm_multi_real = scaler_test.inverse_transform(
    np.concatenate([y_test_lstm_multi_avg.reshape(-1, 1)] + 
                   [np.zeros((len(y_test_lstm_multi_avg), 1)) for _ in range(4)], axis=1)
)[:, close_idx]

y_pred_lstm_multi_real = scaler_test.inverse_transform(
    np.concatenate([y_pred_lstm_multi_avg.reshape(-1, 1)] + 
                   [np.zeros((len(y_pred_lstm_multi_avg), 1)) for _ in range(4)], axis=1)
)[:, close_idx]

# Métricas
mse_lstm_multi = mean_squared_error(y_test_lstm_multi_real, y_pred_lstm_multi_real)
mae_lstm_multi = mean_absolute_error(y_test_lstm_multi_real, y_pred_lstm_multi_real)
r2_lstm_multi = r2_score(y_test_lstm_multi_real, y_pred_lstm_multi_real)

print(f"\n📊 MÉTRICAS LSTM MULTIVARIADO:")
print(f"   MSE: {mse_lstm_multi:.2f}")
print(f"   MAE: {mae_lstm_multi:.2f}")
print(f"   R²:  {r2_lstm_multi:.4f}")

# ============================================================================
# 4. VISUALIZACIÓN
# ============================================================================
print("\n📈 Generando visualización...")

plt.figure(figsize=(14, 6))
plt.plot(y_test_lstm_multi_real, label='Valores Reales', color='blue', linewidth=2)
plt.plot(y_pred_lstm_multi_real, label='Predicciones LSTM Multi', color='purple', linewidth=2, alpha=0.7)
plt.title('Predicciones vs Valores Reales - LSTM Multivariado', fontsize=14, fontweight='bold')
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
print("SECCIÓN 10 COMPLETADA ✅")
print("=" * 80)
print(f"\n🎯 Modelo LSTM Multivariado entrenado:")
print(f"   • Arquitectura: 2 capas LSTM (64→32)")
print(f"   • Input: 3 features (High, Volume, Volatility)")
print(f"   • R²: {r2_lstm_multi:.4f}")
print(f"   • MSE: {mse_lstm_multi:.2f}")
print(f"   • MAE: {mae_lstm_multi:.2f} USD")
print(f"\n📊 Comparación de TODOS los modelos:")
print(f"   • Baseline:     R² = {r2_baseline:.4f}")
print(f"   • LSTM Univ:    R² = {r2_lstm_univ:.4f}")
print(f"   • CNN Univ:     R² = {r2_cnn_univ:.4f}")
print(f"   • LSTM Multi:   R² = {r2_lstm_multi:.4f}")
print(f"\n🏆 Mejor modelo: Baseline (R²={r2_baseline:.4f})")
print(f"\n🎯 Variables en memoria:")
print("   • model_lstm_multi, history_lstm_multi")
print("   • y_pred_lstm_multi_real, y_test_lstm_multi_real")
print("   • mse_lstm_multi, mae_lstm_multi, r2_lstm_multi")
print("\n➡️  Continuar con Sección 11: Comparación de Modelos")
print("=" * 80)
