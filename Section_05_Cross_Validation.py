"""
================================================================================
SECTION 5: CROSS-VALIDATION STRATEGIES
================================================================================
Descripción: Comparar 3 estrategias de validación cruzada temporal
Tiempo estimado: 20 segundos
Outputs esperados: 1 visualización comparando Fixed, Expanding, Sliding CV
================================================================================
NOTA: Ejecutar Sections 01-04 primero
================================================================================
"""

print("\n" + "=" * 80)
print("SECCIÓN 5/13: ESTRATEGIAS DE VALIDACIÓN CRUZADA")
print("=" * 80)

# ============================================================================
# DEFINIR LAS 3 ESTRATEGIAS
# ============================================================================
print("\n[1/2] Definiendo estrategias de validación cruzada...")

# Parámetros
n_splits = 5
total_size = len(df_filtered)
test_size = total_size // (n_splits + 1)

print(f"\nParámetros:")
print(f"  Total de datos: {total_size}")
print(f"  Número de splits: {n_splits}")
print(f"  Tamaño de test por split: {test_size}")

# Estrategia 1: Fixed Split (División Fija)
print(f"\n1. Fixed Split (División Fija)")
print(f"   Train inicial fijo, test se mueve hacia adelante")

# Estrategia 2: Expanding Window (Ventana Expansiva)
print(f"\n2. Expanding Window (Ventana Expansiva)")
print(f"   Train crece con cada split, test se mueve")

# Estrategia 3: Sliding Window (Ventana Deslizante)
print(f"\n3. Sliding Window (Ventana Deslizante)")
print(f"   Train y test se mueven manteniendo tamaño constante")

print("\n✅ Estrategias definidas")

# ============================================================================
# VISUALIZACIÓN COMPARATIVA
# ============================================================================
print("\n[2/2] Generando visualización comparativa...")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))
fig.suptitle('Comparación de Estrategias de Validación Cruzada Temporal', 
             fontsize=14, fontweight='bold')

strategies = ['Fixed Split', 'Expanding Window', 'Sliding Window']
colors_train = ['skyblue', 'lightgreen', 'lightsalmon']
colors_test = ['darkblue', 'darkgreen', 'darkred']

for idx, (strategy, ax) in enumerate(zip(strategies, axes)):
    ax.set_title(strategy, fontsize=12, fontweight='bold')
    ax.set_xlim(0, total_size)
    ax.set_ylim(0, n_splits + 1)
    ax.set_xlabel('Índice de datos')
    ax.set_ylabel('Split')
    ax.invert_yaxis()
    
    for split in range(n_splits):
        y_pos = split + 1
        
        if strategy == 'Fixed Split':
            # Train fijo desde el inicio
            train_start = 0
            train_end = test_size * (split + 1)
            test_start = train_end
            test_end = test_start + test_size
            
        elif strategy == 'Expanding Window':
            # Train crece desde el inicio
            train_start = 0
            train_end = test_size * (split + 2)
            test_start = train_end
            test_end = test_start + test_size
            
        else:  # Sliding Window
            # Train y test se mueven
            train_start = test_size * split
            train_end = train_start + test_size * 2
            test_start = train_end
            test_end = test_start + test_size
        
        # Asegurar que no exceda el tamaño total
        if test_end <= total_size:
            # Dibujar train
            ax.barh(y_pos, train_end - train_start, left=train_start, 
                   height=0.4, color=colors_train[idx], 
                   edgecolor='black', linewidth=0.5, label='Train' if split == 0 else '')
            # Dibujar test
            ax.barh(y_pos, test_end - test_start, left=test_start, 
                   height=0.4, color=colors_test[idx], 
                   edgecolor='black', linewidth=0.5, label='Test' if split == 0 else '')
    
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.show()

print("✅ Visualización comparativa generada")

# ============================================================================
# ANÁLISIS DE ESTRATEGIAS
# ============================================================================
print("\n" + "=" * 80)
print("ANÁLISIS COMPARATIVO")
print("=" * 80)

print("\n1. Fixed Split:")
print("   ✓ Ventajas: Simple, train siempre desde el inicio")
print("   ✗ Desventajas: Test puede ser muy lejano del train inicial")

print("\n2. Expanding Window:")
print("   ✓ Ventajas: Usa toda la información disponible hasta el momento")
print("   ✗ Desventajas: Train crece mucho, puede dar más peso a datos antiguos")

print("\n3. Sliding Window:")
print("   ✓ Ventajas: Mantiene tamaño constante, enfoca en datos recientes")
print("   ✗ Desventajas: Descarta información histórica")

print("\n" + "=" * 80)
print("RECOMENDACIÓN:")
print("Para series temporales financieras, Expanding Window suele ser mejor")
print("porque usa toda la información histórica disponible.")
print("=" * 80)

print("\n✅ SECCIÓN 5 COMPLETADA")
print("=" * 80)
print("Visualización generada:")
print("  - Comparación de 3 estrategias CV")
print("=" * 80)
print("Siguiente: Section_06_Data_Preparation.py")
print("=" * 80)
