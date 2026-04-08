"""
================================================================================
SECTION 3: FEATURE ENGINEERING
================================================================================
Descripción: Crear Daily_Return y Volatility antes del EDA
Tiempo estimado: 5 segundos
Outputs esperados: 2 nuevas columnas, 1036 registros (después de eliminar NaNs)
================================================================================
NOTA: Ejecutar Section_01 y Section_02 primero
================================================================================
"""

print("\n" + "=" * 80)
print("SECCIÓN 3/13: FEATURE ENGINEERING")
print("=" * 80)

# ============================================================================
# CREAR DAILY_RETURN (Retorno Diario)
# ============================================================================
print("\n[1/3] Creando Daily_Return...")

# Calcular retorno diario porcentual
df_filtered['Daily_Return'] = df_filtered['Close'].pct_change() * 100

print(f"✅ Daily_Return creado")
print(f"   Fórmula: (Close_t - Close_t-1) / Close_t-1 * 100")
print(f"   Ejemplo: {df_filtered['Daily_Return'].dropna().head(3).values}")

# ============================================================================
# CREAR VOLATILITY (Volatilidad con ventana de 20 días)
# ============================================================================
print("\n[2/3] Creando Volatility...")

# Calcular volatilidad como desviación estándar móvil de 20 días del retorno diario
df_filtered['Volatility'] = df_filtered['Daily_Return'].rolling(window=20).std()

print(f"✅ Volatility creado")
print(f"   Fórmula: std(Daily_Return, window=20)")
print(f"   Ventana: 20 días")

# ============================================================================
# ELIMINAR VALORES NaN
# ============================================================================
print("\n[3/3] Eliminando valores NaN...")

# Guardar tamaño original
original_size = len(df_filtered)

# Eliminar NaNs generados por pct_change() y rolling()
df_filtered = df_filtered.dropna().reset_index(drop=True)

print(f"✅ NaNs eliminados")
print(f"   Registros originales: {original_size}")
print(f"   Registros después de limpieza: {len(df_filtered)}")
print(f"   Registros eliminados: {original_size - len(df_filtered)}")

# ============================================================================
# VERIFICACIÓN DE FEATURES
# ============================================================================
print("\n" + "=" * 80)
print("FEATURES CREADOS")
print("=" * 80)

print(f"\nColumnas actuales: {list(df_filtered.columns)}")
print(f"Total de variables: {len(df_filtered.columns)}")

print(f"\n1. Close (Precio de cierre)")
print(f"   Rango: ${df_filtered['Close'].min():.2f} - ${df_filtered['Close'].max():.2f}")

print(f"\n2. High (Precio máximo)")
print(f"   Rango: ${df_filtered['High'].min():.2f} - ${df_filtered['High'].max():.2f}")

print(f"\n3. Volume (Volumen)")
print(f"   Rango: {df_filtered['Volume'].min():.0f} - {df_filtered['Volume'].max():.0f}")

print(f"\n4. Daily_Return (Retorno diario %)")
print(f"   Rango: {df_filtered['Daily_Return'].min():.2f}% - {df_filtered['Daily_Return'].max():.2f}%")
print(f"   Media: {df_filtered['Daily_Return'].mean():.2f}%")

print(f"\n5. Volatility (Volatilidad)")
print(f"   Rango: {df_filtered['Volatility'].min():.2f} - {df_filtered['Volatility'].max():.2f}")
print(f"   Media: {df_filtered['Volatility'].mean():.2f}")

# Mostrar primeras filas
print(f"\nPrimeras 5 filas con features:")
print(df_filtered.head())

print("\n✅ SECCIÓN 3 COMPLETADA")
print("=" * 80)
print("Siguiente: Section_04_EDA.py")
print("=" * 80)
