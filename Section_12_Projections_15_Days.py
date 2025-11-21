# ================================================================================
# SECCIÓN 12/13: PROYECCIONES A 15 DÍAS
# ================================================================================
# 
# REQUISITOS: Haber ejecutado las secciones 1-11 previamente
# TIEMPO ESTIMADO: 1-2 minutos
# 
# OUTPUTS:
# - 4 gráficas individuales de proyecciones (una por modelo)
# - 1 gráfica comparativa de todas las proyecciones
# - CSV con proyecciones de todos los modelos
# ================================================================================

print("\n" + "="*80)
print("SECCIÓN 12/13: PROYECCIONES A 15 DÍAS")
print("="*80)

# Importaciones necesarias
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Parámetros
PROJECTION_DAYS = 15

# ================================================================================
# PASO 1: Función de Proyección Iterativa
# ================================================================================
print("\n[1/5] Definiendo función de proyección iterativa...")

def project_future(model, initial_sequence, scaler, n_days, model_type='baseline'):
    """
    Proyecta n_days hacia el futuro usando predicción iterativa
    
    Args:
        model: Modelo entrenado
        initial_sequence: Secuencia inicial (últimos 30 días del test set)
        scaler: Scaler usado para entrenar el modelo (fit on 5 features)
        n_days: Número de días a proyectar
        model_type: 'baseline', 'lstm_univ', 'cnn_univ', 'lstm_multi'
    
    Returns:
        Array con proyecciones en escala original
    """
    projections = []
    current_sequence = initial_sequence.copy()
    
    for _ in range(n_days):
        if model_type == 'baseline':
            # Para baseline, usar solo el último valor
            pred_scaled = model.predict(current_sequence[-1:].reshape(1, -1))[0]
        elif model_type in ['lstm_univ', 'cnn_univ']:
            # Para modelos univariados
            pred_scaled = model.predict(current_sequence.reshape(1, TIMESTEPS, 1), verbose=0)[0, 0]
        else:  # lstm_multi
            # Para modelo multivariado (solo proyectamos Close, pero necesitamos 3 features)
            # Usar valores dummy para High y Volume
            current_multi = np.zeros((1, TIMESTEPS, 3))
            current_multi[0, :, 0] = current_sequence.flatten()  # Close
            current_multi[0, :, 1] = current_sequence.flatten() * 1.01  # High dummy
            current_multi[0, :, 2] = 1000000  # Volume dummy
            pred_scaled = model.predict(current_multi, verbose=0)[0, 0]
        
        # Invertir escala: scaler tiene 5 features [Close, High, Volume, Daily_Return, Volatility]
        # Close está en índice 0
        dummy_row = np.zeros((1, 5))  # Crear fila con 5 features
        dummy_row[0, 0] = pred_scaled  # Poner predicción en Close (índice 0)
        pred_original = scaler.inverse_transform(dummy_row)[0, 0]  # Extraer Close
        projections.append(pred_original)
        
        # Actualizar secuencia (agregar predicción y quitar valor más antiguo)
        # Escalar la predicción para usarla en próxima iteración
        dummy_row_forward = np.zeros((1, 5))
        dummy_row_forward[0, 0] = pred_original
        new_val_scaled = scaler.transform(dummy_row_forward)[0, 0]
        current_sequence = np.append(current_sequence[1:], new_val_scaled)
    
    return np.array(projections)

print("✅ Función de proyección definida")

# ================================================================================
# PASO 2: Generar Proyecciones para Todos los Modelos
# ================================================================================
print("\n[2/5] Generando proyecciones para todos los modelos...")

# Obtener última secuencia del test set (últimos 30 días)
last_sequence_scaled = X_test_univ[-1].flatten()

# Proyecciones para cada modelo
print("   - Proyectando Baseline...")
proj_baseline = project_future(baseline_model, last_sequence_scaled, scaler_test, 
                               PROJECTION_DAYS, model_type='baseline')

print("   - Proyectando LSTM Univariado...")
proj_lstm_univ = project_future(lstm_univ_model, last_sequence_scaled, scaler_test, 
                                PROJECTION_DAYS, model_type='lstm_univ')

print("   - Proyectando CNN Univariado...")
proj_cnn_univ = project_future(cnn_univ_model, last_sequence_scaled, scaler_test, 
                               PROJECTION_DAYS, model_type='cnn_univ')

print("   - Proyectando LSTM Multivariado...")
proj_lstm_multi = project_future(lstm_multi_model, last_sequence_scaled, scaler_test, 
                                 PROJECTION_DAYS, model_type='lstm_multi')

print(f"✅ Proyecciones generadas: {PROJECTION_DAYS} días hacia el futuro")

# ================================================================================
# PASO 3: Crear DataFrame con Proyecciones
# ================================================================================
print("\n[3/5] Creando DataFrame con proyecciones...")

# Crear fechas futuras (aproximadas)
last_date_idx = len(y_test_univ) - 1
future_indices = np.arange(last_date_idx + 1, last_date_idx + 1 + PROJECTION_DAYS)

projections_df = pd.DataFrame({
    'Día': range(1, PROJECTION_DAYS + 1),
    'Baseline': proj_baseline,
    'LSTM_Univariado': proj_lstm_univ,
    'CNN_Univariado': proj_cnn_univ,
    'LSTM_Multivariado': proj_lstm_multi
})

print("\n" + "="*80)
print("PROYECCIONES A 15 DÍAS (USD)")
print("="*80)
print(projections_df.to_string(index=False))
print("="*80)

# ================================================================================
# PASO 4: Visualizar Proyecciones Individuales
# ================================================================================
print("\n[4/5] Generando visualizaciones individuales...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Proyecciones a 15 Días - Individual por Modelo', fontsize=16, fontweight='bold')

models_data = [
    ('Baseline', proj_baseline, 'gray', axes[0, 0]),
    ('LSTM Univariado', proj_lstm_univ, 'green', axes[0, 1]),
    ('CNN Univariado', proj_cnn_univ, 'red', axes[1, 0]),
    ('LSTM Multivariado', proj_lstm_multi, 'purple', axes[1, 1])
]

for model_name, projections, color, ax in models_data:
    # Últimos 30 días del test set
    last_30_days = y_test_univ[-30:]
    
    # Combinar últimos días reales con proyecciones
    combined_x = np.arange(-30, PROJECTION_DAYS)
    combined_y = np.concatenate([last_30_days, projections])
    
    # Graficar
    ax.plot(range(-30, 0), last_30_days, color='blue', linewidth=2, label='Histórico (Test)', alpha=0.7)
    ax.plot(range(0, PROJECTION_DAYS), projections, color=color, linewidth=2, 
            linestyle='--', marker='o', markersize=4, label='Proyección')
    ax.axvline(x=0, color='black', linestyle=':', linewidth=1, alpha=0.5)
    
    ax.set_title(f'{model_name}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Días (0 = hoy)', fontsize=10)
    ax.set_ylabel('Precio Close (USD)', fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
print("✅ Gráficas individuales generadas")
plt.show()

# ================================================================================
# PASO 5: Visualización Comparativa
# ================================================================================
print("\n[5/5] Generando visualización comparativa...")

plt.figure(figsize=(15, 8))

# Últimos 50 días del test set para contexto
context_days = 50
last_days = y_test_univ[-context_days:]

# Graficar histórico
plt.plot(range(-context_days, 0), last_days, color='blue', linewidth=2.5, 
         label='Histórico (Test)', alpha=0.8)

# Graficar proyecciones de cada modelo
plt.plot(range(0, PROJECTION_DAYS), proj_baseline, color='gray', linewidth=2, 
         linestyle='--', marker='o', markersize=5, label='Baseline', alpha=0.7)
plt.plot(range(0, PROJECTION_DAYS), proj_lstm_univ, color='green', linewidth=2, 
         linestyle='--', marker='s', markersize=5, label='LSTM Univariado', alpha=0.7)
plt.plot(range(0, PROJECTION_DAYS), proj_cnn_univ, color='red', linewidth=2, 
         linestyle='--', marker='^', markersize=5, label='CNN Univariado', alpha=0.7)
plt.plot(range(0, PROJECTION_DAYS), proj_lstm_multi, color='purple', linewidth=2, 
         linestyle='--', marker='d', markersize=5, label='LSTM Multivariado', alpha=0.7)

# Línea divisoria
plt.axvline(x=0, color='black', linestyle=':', linewidth=2, alpha=0.5, label='Hoy')

# Configurar gráfica
plt.title('Proyecciones a 15 Días - Comparación de Todos los Modelos', fontsize=16, fontweight='bold')
plt.xlabel('Días (0 = hoy)', fontsize=12)
plt.ylabel('Precio Close (USD)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()

print("✅ Gráfica comparativa generada")
plt.show()

# ================================================================================
# RESUMEN DE PROYECCIONES
# ================================================================================
print("\n" + "="*80)
print("RESUMEN DE PROYECCIONES")
print("="*80)

print(f"\n📅 Proyecciones a {PROJECTION_DAYS} días desde hoy:")
print(f"\nÚltimo precio conocido (test set): ${y_test_univ[-1]:.2f} USD")

for model_name, projections in [('Baseline', proj_baseline), 
                                 ('LSTM Univariado', proj_lstm_univ),
                                 ('CNN Univariado', proj_cnn_univ),
                                 ('LSTM Multivariado', proj_lstm_multi)]:
    day_15_price = projections[-1]
    price_change = day_15_price - y_test_univ[-1]
    pct_change = (price_change / y_test_univ[-1]) * 100
    
    print(f"\n{model_name}:")
    print(f"  - Precio día 15: ${day_15_price:.2f} USD")
    print(f"  - Cambio: ${price_change:+.2f} USD ({pct_change:+.2f}%)")
    
    if pct_change > 10:
        print(f"  - Tendencia: 📈 Fuerte alza")
    elif pct_change > 3:
        print(f"  - Tendencia: ↗️ Alza moderada")
    elif pct_change > -3:
        print(f"  - Tendencia: ➡️ Estable")
    elif pct_change > -10:
        print(f"  - Tendencia: ↘️ Baja moderada")
    else:
        print(f"  - Tendencia: 📉 Fuerte baja")

print("\n" + "="*80)
print("✅ SECCIÓN 12 COMPLETADA")
print("="*80)
print("\nPróxima sección: Guardar resultados (Section_13_Save_Results.py)")
