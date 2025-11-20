"""
================================================================================
SECTION 4: EXPLORATORY DATA ANALYSIS (EDA)
================================================================================
Descripción: Análisis exploratorio de 5 variables con visualizaciones
Tiempo estimado: 30-60 segundos
Outputs esperados: 2 visualizaciones (series temporales + correlación)
================================================================================
NOTA: Ejecutar Sections 01, 02, 03 primero
================================================================================
"""

print("\n" + "=" * 80)
print("SECCIÓN 4/13: ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("=" * 80)

# ============================================================================
# ESTADÍSTICAS DESCRIPTIVAS DE 5 VARIABLES
# ============================================================================
print("\n[1/3] Calculando estadísticas descriptivas...")

features_to_analyze = ['Close', 'High', 'Volume', 'Daily_Return', 'Volatility']

print("\n" + "=" * 80)
print("ESTADÍSTICAS DESCRIPTIVAS (5 VARIABLES)")
print("=" * 80)

for feature in features_to_analyze:
    if feature in df_filtered.columns:
        stats = df_filtered[feature].describe()
        print(f"\n{feature}:")
        print(f"  count: {stats.iloc[0]:.0f}")
        print(f"  mean:  {stats.iloc[1]:.6f}")
        print(f"  std:   {stats.iloc[2]:.6f}")
        print(f"  min:   {stats.iloc[3]:.6f}")
        print(f"  25%:   {stats.iloc[4]:.6f}")
        print(f"  50%:   {stats.iloc[5]:.6f}")
        print(f"  75%:   {stats.iloc[6]:.6f}")
        print(f"  max:   {stats.iloc[7]:.6f}")

print("\n✅ Estadísticas calculadas para 5 variables")

# ============================================================================
# VISUALIZACIÓN 1: SERIES TEMPORALES (5 SUBPLOTS)
# ============================================================================
print("\n[2/3] Generando gráfico de series temporales...")

fig, axes = plt.subplots(5, 1, figsize=(15, 12))
fig.suptitle('Series Temporales - 5 Variables', fontsize=16, fontweight='bold')

# Plot para cada variable
for idx, feature in enumerate(features_to_analyze):
    axes[idx].plot(df_filtered['Date'], df_filtered[feature], linewidth=1.5)
    axes[idx].set_title(f'{feature}', fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('Fecha')
    axes[idx].set_ylabel(feature)
    axes[idx].grid(True, alpha=0.3)
    axes[idx].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

print("✅ Gráfico de series temporales generado (5 subplots)")

# ============================================================================
# VISUALIZACIÓN 2: MATRIZ DE CORRELACIÓN (5x5)
# ============================================================================
print("\n[3/3] Generando matriz de correlación...")

# Calcular correlaciones
correlation_matrix = df_filtered[features_to_analyze].corr()

# Crear heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, 
            annot=True, 
            fmt='.3f', 
            cmap='coolwarm', 
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8})
plt.title('Matriz de Correlación - 5 Variables', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.show()

print("✅ Matriz de correlación generada (5x5)")

# ============================================================================
# ANÁLISIS DE CORRELACIONES
# ============================================================================
print("\n" + "=" * 80)
print("ANÁLISIS DE CORRELACIONES")
print("=" * 80)

print("\nCorrelaciones con Close (variable objetivo):")
close_corr = correlation_matrix['Close'].sort_values(ascending=False)
for feature, corr in close_corr.items():
    if feature != 'Close':
        print(f"  {feature:20s}: {corr:+.3f}")

# Encontrar la correlación más fuerte (excluyendo Close consigo mismo)
close_corr_filtered = close_corr.drop('Close')
strongest_corr = close_corr_filtered.abs().idxmax()
strongest_value = close_corr_filtered[strongest_corr]

print(f"\nVariable más correlacionada con Close:")
print(f"  {strongest_corr} → {strongest_value:+.3f}")

print("\n✅ SECCIÓN 4 COMPLETADA")
print("=" * 80)
print("Visualizaciones generadas:")
print("  1. Series temporales (5 variables)")
print("  2. Matriz de correlación (5x5)")
print("=" * 80)
print("Siguiente: Section_05_Cross_Validation.py")
print("=" * 80)
