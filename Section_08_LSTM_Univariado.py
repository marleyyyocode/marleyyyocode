"""
================================================================================
SECCIÓN 8/13: LSTM UNIVARIADO
================================================================================
Entrena modelo LSTM con entrada univariada (solo Close).

Arquitectura: 2 capas LSTM (64→32 unidades)
Hiperparámetros: LR=0.001, Epochs=100, Batch=32, No early stopping
Tiempo estimado: 8-12 minutos
Resultado esperado: R² ≈ 0.948
"""

print("=" * 80)
print("SECCIÓN 8/13: LSTM UNIVARIADO")
print("=" * 80)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# ============================================================================
# 1. CONSTRUIR MODELO LSTM
# ============================================================================
print("\n[1/3] Construyendo arquitectura LSTM...")

model_lstm_univ = Sequential([
    LSTM(64, return_sequences=True, input_shape=(TIMESTEPS, 1)),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(HORIZON)
], name='LSTM_Univariado')

model_lstm_univ.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

print(f"✅ Modelo LSTM Univariado construido")
model_lstm_univ.summary()

# ============================================================================
# 2. ENTRENAR MODELO
# ============================================================================
print("\n[2/3] Entrenando LSTM Univariado (100 épocas, sin early stopping)...")
print("⏳ Esto tomará aproximadamente 8-12 minutos...")

history_lstm_univ = model_lstm_univ.fit(
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
y_pred_lstm_univ = model_lstm_univ.predict(X_test_univ)

# Usar promedio de predicciones multi-step
y_pred_lstm_univ_avg = y_pred_lstm_univ.mean(axis=1)
y_test_lstm_univ_avg = y_test_univ.mean(axis=1)

# Des-escalar
# Close es la primera columna (índice 0) en la lista de features_scale
close_idx = 0  # 'Close' es la primera columna

y_test_lstm_real = scaler_test.inverse_transform(
    np.concatenate([y_test_lstm_univ_avg.reshape(-1, 1)] + 
                   [np.zeros((len(y_test_lstm_univ_avg), 1)) for _ in range(4)], axis=1)
)[:, close_idx]

y_pred_lstm_real = scaler_test.inverse_transform(
    np.concatenate([y_pred_lstm_univ_avg.reshape(-1, 1)] + 
                   [np.zeros((len(y_pred_lstm_univ_avg), 1)) for _ in range(4)], axis=1)
)[:, close_idx]

# Métricas
mse_lstm_univ = mean_squared_error(y_test_lstm_real, y_pred_lstm_real)
mae_lstm_univ = mean_absolute_error(y_test_lstm_real, y_pred_lstm_real)
r2_lstm_univ = r2_score(y_test_lstm_real, y_pred_lstm_real)

print(f"\n📊 MÉTRICAS LSTM UNIVARIADO:")
print(f"   MSE: {mse_lstm_univ:.2f}")
print(f"   MAE: {mae_lstm_univ:.2f}")
print(f"   R²:  {r2_lstm_univ:.4f}")

# ============================================================================
# 4. VISUALIZACIÓN
# ============================================================================
print("\n📈 Generando visualización...")

plt.figure(figsize=(14, 6))
plt.plot(y_test_lstm_real, label='Valores Reales', color='blue', linewidth=2)
plt.plot(y_pred_lstm_real, label='Predicciones LSTM', color='green', linewidth=2, alpha=0.7)
plt.title('Predicciones vs Valores Reales - LSTM Univariado', fontsize=14, fontweight='bold')
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
print("SECCIÓN 8 COMPLETADA ✅")
print("=" * 80)
print(f"\n🎯 Modelo LSTM Univariado entrenado:")
print(f"   • Arquitectura: 2 capas LSTM (64→32)")
print(f"   • R²: {r2_lstm_univ:.4f}")
print(f"   • MSE: {mse_lstm_univ:.2f}")
print(f"   • MAE: {mae_lstm_univ:.2f} USD")
print(f"\n📊 Comparación con Baseline:")
print(f"   • Baseline R²: {r2_baseline:.4f}")
print(f"   • LSTM R²:     {r2_lstm_univ:.4f}")
print(f"   • Diferencia:  {(r2_lstm_univ - r2_baseline):.4f}")
print(f"\n🎯 Variables en memoria:")
print("   • model_lstm_univ, history_lstm_univ")
print("   • y_pred_lstm_real, y_test_lstm_real")
print("   • mse_lstm_univ, mae_lstm_univ, r2_lstm_univ")
print("\n➡️  Continuar con Sección 9: CNN Univariado")
print("=" * 80)
