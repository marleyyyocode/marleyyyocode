"""
================================================================================
SECCIÓN 6/13: PREPARACIÓN DE DATOS
================================================================================
Prepara los datos para entrenamiento:
- División temporal (Train/Val/Test)
- Escalado con MinMaxScaler
- Creación de secuencias para modelos
- Secuencias univariadas y multivariadas

Tiempo estimado: 30 segundos
"""

print("=" * 80)
print("SECCIÓN 6/13: PREPARACIÓN DE DATOS")
print("=" * 80)

# ============================================================================
# 1. DIVISIÓN TEMPORAL DE DATOS
# ============================================================================
print("\n[1/4] División temporal de datos...")

# Fechas de corte
train_end = pd.Timestamp('2023-11-30')
val_end = pd.Timestamp('2024-02-28')

# División
df_train = df_filtered[df_filtered['Date'] <= train_end].copy()
df_val = df_filtered[(df_filtered['Date'] > train_end) & (df_filtered['Date'] <= val_end)].copy()
df_test = df_filtered[df_filtered['Date'] > val_end].copy()

print(f"✅ Train: {len(df_train)} registros ({df_train['Date'].min()} a {df_train['Date'].max()})")
print(f"✅ Val:   {len(df_val)} registros ({df_val['Date'].min()} a {df_val['Date'].max()})")
print(f"✅ Test:  {len(df_test)} registros ({df_test['Date'].min()} a {df_test['Date'].max()})")

# ============================================================================
# 2. ESCALADO DE DATOS
# ============================================================================
print("\n[2/4] Escalado de datos con MinMaxScaler...")

from sklearn.preprocessing import MinMaxScaler

# Columnas a escalar
features_scale = ['Close', 'High', 'Volume', 'Daily_Return', 'Volatility']

# Crear escaladores
scaler_train = MinMaxScaler()
scaler_val = MinMaxScaler()
scaler_test = MinMaxScaler()

# Ajustar y transformar cada conjunto
df_train_scaled = df_train.copy()
df_train_scaled[features_scale] = scaler_train.fit_transform(df_train[features_scale])

df_val_scaled = df_val.copy()
df_val_scaled[features_scale] = scaler_val.fit_transform(df_val[features_scale])

df_test_scaled = df_test.copy()
df_test_scaled[features_scale] = scaler_test.fit_transform(df_test[features_scale])

print(f"✅ Datos escalados (rango 0-1)")
print(f"   Train: {df_train_scaled.shape}")
print(f"   Val:   {df_val_scaled.shape}")
print(f"   Test:  {df_test_scaled.shape}")

# ============================================================================
# 3. CREACIÓN DE SECUENCIAS UNIVARIADAS (CLOSE)
# ============================================================================
print("\n[3/4] Creando secuencias univariadas (Close)...")

TIMESTEPS = 30
HORIZON = 5

def create_sequences_univariate(data, timesteps, horizon):
    """Crea secuencias para modelos univariados (solo Close)"""
    X, y = [], []
    close_col = 'Close'
    
    for i in range(len(data) - timesteps - horizon + 1):
        # X: últimos 'timesteps' valores
        X.append(data[close_col].iloc[i:i+timesteps].values)
        # y: próximos 'horizon' valores
        y.append(data[close_col].iloc[i+timesteps:i+timesteps+horizon].values)
    
    return np.array(X), np.array(y)

# Crear secuencias univariadas
X_train_univ, y_train_univ = create_sequences_univariate(df_train_scaled, TIMESTEPS, HORIZON)
X_val_univ, y_val_univ = create_sequences_univariate(df_val_scaled, TIMESTEPS, HORIZON)
X_test_univ, y_test_univ = create_sequences_univariate(df_test_scaled, TIMESTEPS, HORIZON)

# Reshape para LSTM/CNN (samples, timesteps, features)
X_train_univ = X_train_univ.reshape((X_train_univ.shape[0], X_train_univ.shape[1], 1))
X_val_univ = X_val_univ.reshape((X_val_univ.shape[0], X_val_univ.shape[1], 1))
X_test_univ = X_test_univ.reshape((X_test_univ.shape[0], X_test_univ.shape[1], 1))

print(f"✅ Secuencias univariadas creadas:")
print(f"   X_train_univ: {X_train_univ.shape} -> y_train_univ: {y_train_univ.shape}")
print(f"   X_val_univ:   {X_val_univ.shape} -> y_val_univ:   {y_val_univ.shape}")
print(f"   X_test_univ:  {X_test_univ.shape} -> y_test_univ:  {y_test_univ.shape}")

# ============================================================================
# 4. CREACIÓN DE SECUENCIAS MULTIVARIADAS
# ============================================================================
print("\n[4/4] Creando secuencias multivariadas (High, Volume, Volatility)...")

def create_sequences_multivariate(data, timesteps, horizon):
    """Crea secuencias para modelo multivariado"""
    X, y = [], []
    
    # Features de entrada: High, Volume, Volatility
    input_features = ['High', 'Volume', 'Volatility']
    # Target: Close
    target_feature = 'Close'
    
    for i in range(len(data) - timesteps - horizon + 1):
        # X: últimos 'timesteps' valores de las 3 features
        X.append(data[input_features].iloc[i:i+timesteps].values)
        # y: próximos 'horizon' valores de Close
        y.append(data[target_feature].iloc[i+timesteps:i+timesteps+horizon].values)
    
    return np.array(X), np.array(y)

# Crear secuencias multivariadas
X_train_multi, y_train_multi = create_sequences_multivariate(df_train_scaled, TIMESTEPS, HORIZON)
X_val_multi, y_val_multi = create_sequences_multivariate(df_val_scaled, TIMESTEPS, HORIZON)
X_test_multi, y_test_multi = create_sequences_multivariate(df_test_scaled, TIMESTEPS, HORIZON)

print(f"✅ Secuencias multivariadas creadas:")
print(f"   X_train_multi: {X_train_multi.shape} -> y_train_multi: {y_train_multi.shape}")
print(f"   X_val_multi:   {X_val_multi.shape} -> y_val_multi:   {y_val_multi.shape}")
print(f"   X_test_multi:  {X_test_multi.shape} -> y_test_multi:  {y_test_multi.shape}")

# ============================================================================
# RESUMEN
# ============================================================================
print("\n" + "=" * 80)
print("SECCIÓN 6 COMPLETADA ✅")
print("=" * 80)
print("\n📊 Datos preparados:")
print(f"   • Conjuntos: Train ({len(df_train)}), Val ({len(df_val)}), Test ({len(df_test)})")
print(f"   • Secuencias univariadas: {X_train_univ.shape[0]} train, {X_test_univ.shape[0]} test")
print(f"   • Secuencias multivariadas: {X_train_multi.shape[0]} train, {X_test_multi.shape[0]} test")
print(f"   • Timesteps: {TIMESTEPS}, Horizon: {HORIZON}")
print("\n🎯 Variables en memoria:")
print("   • df_train_scaled, df_val_scaled, df_test_scaled")
print("   • X_train_univ, y_train_univ, X_test_univ, y_test_univ (univariadas)")
print("   • X_train_multi, y_train_multi, X_test_multi, y_test_multi (multivariadas)")
print("   • scaler_train, scaler_val, scaler_test")
print("\n➡️  Continuar con Sección 7: Baseline Model")
print("=" * 80)
