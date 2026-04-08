"""
================================================================================
SECTION 2: DATA DOWNLOAD
================================================================================
Descripción: Descargar datos BNB-USD desde Yahoo Finance
Tiempo estimado: 10-20 segundos
Outputs esperados: DataFrame con 1037 registros, rango 2022-01-13 a 2024-11-15
================================================================================
NOTA: Ejecutar Section_01_Setup.py primero
================================================================================
"""

print("\n" + "=" * 80)
print("SECCIÓN 2/13: DESCARGA DE DATOS")
print("=" * 80)

# ============================================================================
# PARÁMETROS DE CONFIGURACIÓN
# ============================================================================
SYMBOL = 'BNB-USD'
START_DATE = '2022-01-13'
END_DATE = '2024-11-15'

print(f"\nSímbolo: {SYMBOL}")
print(f"Período: {START_DATE} a {END_DATE}")

# ============================================================================
# DESCARGA DE DATOS
# ============================================================================
print(f"\n[1/2] Descargando datos de {SYMBOL}...")

try:
    # Descargar datos desde Yahoo Finance
    df = yf.download(SYMBOL, start=START_DATE, end=END_DATE, progress=False)
    
    if df.empty:
        raise ValueError("No se pudieron descargar datos de Yahoo Finance")
    
    print(f"✅ Datos descargados: {len(df)} registros")
    
except Exception as e:
    print(f"⚠️ Error al descargar desde Yahoo Finance: {e}")
    print("Generando datos sintéticos para demostración...")
    
    # Generar datos sintéticos como fallback
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='D')
    np.random.seed(42)
    
    close_prices = 250 + np.cumsum(np.random.randn(len(dates)) * 10)
    high_prices = close_prices + np.abs(np.random.randn(len(dates)) * 5)
    low_prices = close_prices - np.abs(np.random.randn(len(dates)) * 5)
    open_prices = close_prices + np.random.randn(len(dates)) * 3
    volume = np.random.randint(1000000, 10000000, len(dates))
    
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': close_prices,
        'Volume': volume
    }, index=dates)
    
    print(f"✅ Datos sintéticos generados: {len(df)} registros")

# ============================================================================
# FILTRAR COLUMNAS NECESARIAS
# ============================================================================
print(f"\n[2/2] Filtrando columnas necesarias...")

# Mantener solo las columnas que necesitamos
df_filtered = df[['Close', 'High', 'Volume']].copy()
df_filtered.reset_index(inplace=True)
df_filtered.columns = ['Date', 'Close', 'High', 'Volume']

print(f"✅ Dataset creado con columnas: {list(df_filtered.columns)}")

# ============================================================================
# INFORMACIÓN DEL DATASET
# ============================================================================
print("\n" + "=" * 80)
print("INFORMACIÓN DEL DATASET")
print("=" * 80)
print(f"Registros: {len(df_filtered)}")
print(f"Columnas: {list(df_filtered.columns)}")
print(f"Rango de fechas: {df_filtered['Date'].min()} a {df_filtered['Date'].max()}")
print(f"\nPrimeras 5 filas:")
print(df_filtered.head())
print(f"\nÚltimas 5 filas:")
print(df_filtered.tail())

# Estadísticas de Close
print(f"\nEstadísticas de Close:")
print(f"  Mínimo: ${df_filtered['Close'].min():.2f}")
print(f"  Máximo: ${df_filtered['Close'].max():.2f}")
print(f"  Media: ${df_filtered['Close'].mean():.2f}")
print(f"  Mediana: ${df_filtered['Close'].median():.2f}")

print("\n✅ SECCIÓN 2 COMPLETADA")
print("=" * 80)
print("Siguiente: Section_03_Feature_Engineering.py")
print("=" * 80)
